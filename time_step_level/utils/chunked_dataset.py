"""
Chunked Dataset Manager for handling large EEG datasets efficiently.

This module provides classes for:
1. Saving processed data in manageable chunks to avoid RAM overflow
2. Loading data incrementally during training/evaluation
3. Maintaining metadata about the dataset structure
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_process_rss_mb() -> float:
    """Return process RSS memory in MB, using psutil if available."""
    if _PSUTIL_AVAILABLE:
        return _psutil.Process().memory_info().rss / (1024 ** 2)
    # Fallback: read /proc/self/status on Linux
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024  # kB -> MB
    except OSError:
        pass
    return float('nan')


class ChunkedDatasetWriter:
    """
    Handles writing large datasets in chunks to disk.
    
    Each chunk is saved as a separate .npy file with a maximum size limit.
    Metadata is maintained in a JSON file for easy loading later.
    
    Args:
        base_path: Directory where chunks will be saved
        dataset_name: Name prefix for the dataset
        max_chunk_size_gb: Maximum size of each chunk in GB (default: 4GB)
        channels: Number of EEG channels
        window_size: Size of each time window
    """
    
    def __init__(
        self, 
        base_path: str, 
        dataset_name: str,
        max_chunk_size_gb: float = 4.0,
        channels: int = 18,
        window_size: int = 15360
    ):
        self.base_path = Path(base_path)
        self.dataset_name = dataset_name
        self.max_chunk_size_bytes = int(max_chunk_size_gb * 1024**3)
        self.channels = channels
        self.window_size = window_size
        
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Current chunk buffers
        self.data_buffer = []
        self.label_buffer = []
        self.current_buffer_size = 0
        self.chunk_index = 0
        
        # Metadata tracking
        self.chunk_info = []
        self.total_samples = 0

        # Log buffer size every this many samples (0 = disabled)
        self.log_interval = 4000
        
        logger.info(f"Initialized ChunkedDatasetWriter at {self.base_path}")
        logger.info(f"Max chunk size: {max_chunk_size_gb:.2f} GB")
    
    def _get_chunk_path(self, chunk_idx: int, data_type: str = 'data') -> Path:
        """Generate file path for a chunk."""
        return self.base_path / f"{self.dataset_name}_chunk{chunk_idx:04d}_{data_type}.npy"
    
    def _estimate_array_size(self, array: np.ndarray) -> int:
        """Estimate memory size of numpy array in bytes."""
        return array.nbytes
    
    def add_sample(self, data: np.ndarray, label: np.ndarray):
        """
        Add a single sample to the dataset.
        
        Args:
            data: EEG data array, shape (channels, window_size)
            label: Label array, shape (window_size,)
        """
        # Estimate size of new sample
        sample_size = self._estimate_array_size(data) + self._estimate_array_size(label)
        
        # Check if adding this sample would exceed chunk size
        if self.current_buffer_size + sample_size > self.max_chunk_size_bytes and len(self.data_buffer) > 0:
            self._flush_chunk()
        
        # Add to buffer — force copies to release any numpy view references
        # to source tensors (prevents backing batch arrays from being kept alive)
        self.data_buffer.append(np.array(data, dtype=np.float32, copy=True))
        self.label_buffer.append(np.array(label, dtype=np.float32, copy=True))
        self.current_buffer_size += sample_size
        self.total_samples += 1

        if self.log_interval > 0 and self.total_samples % self.log_interval == 0:
            buf_mb = self.current_buffer_size / (1024 ** 2)
            rss_mb = _get_process_rss_mb()
            logger.debug(
                f"[{self.dataset_name}] samples={self.total_samples} "
                f"buffer={buf_mb:.1f} MB "
                f"({len(self.data_buffer)} items) "
                f"process_rss={rss_mb:.0f} MB"
            )
    
    def add_batch(self, data_list: List[np.ndarray], label_list: List[np.ndarray]):
        """
        Add multiple samples at once.
        
        Args:
            data_list: List of EEG data arrays
            label_list: List of label arrays
        """
        for data, label in zip(data_list, label_list):
            self.add_sample(data, label)
    
    def _flush_chunk(self):
        """Save current buffer to disk as a chunk."""
        if len(self.data_buffer) == 0:
            return
        
        # Convert lists to numpy arrays
        data_array = np.array(self.data_buffer, dtype=np.float32)
        label_array = np.array(self.label_buffer, dtype=np.float32)
        
        # Save to disk
        data_path = self._get_chunk_path(self.chunk_index, 'data')
        label_path = self._get_chunk_path(self.chunk_index, 'label')
        
        np.save(data_path, data_array)
        np.save(label_path, label_array)
        
        # Store metadata
        chunk_metadata = {
            'chunk_index': self.chunk_index,
            'num_samples': len(self.data_buffer),
            'data_path': str(data_path),
            'label_path': str(label_path),
            'data_shape': data_array.shape,
            'label_shape': label_array.shape,
            'size_mb': (data_array.nbytes + label_array.nbytes) / (1024**2)
        }
        self.chunk_info.append(chunk_metadata)
        
        logger.info(f"Saved chunk {self.chunk_index}: {len(self.data_buffer)} samples "
                   f"({chunk_metadata['size_mb']:.2f} MB)")
        
        # Clear buffers
        self.data_buffer = []
        self.label_buffer = []
        self.current_buffer_size = 0
        self.chunk_index += 1
    
    def finalize(self) -> Path:
        """
        Finish writing dataset and save metadata.
        
        Returns:
            Path to the metadata JSON file
        """
        # Flush any remaining data
        if len(self.data_buffer) > 0:
            self._flush_chunk()
        
        # Save metadata
        metadata = {
            'dataset_name': self.dataset_name,
            'total_samples': self.total_samples,
            'num_chunks': self.chunk_index,
            'channels': self.channels,
            'window_size': self.window_size,
            'chunks': self.chunk_info
        }
        
        metadata_path = self.base_path / f"{self.dataset_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset finalized: {self.total_samples} samples in {self.chunk_index} chunks")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return metadata_path


class ChunkedDataset(Dataset):
    """
    PyTorch Dataset that loads data from chunked files on-the-fly.
    
    This dataset loads chunks into memory only when needed, significantly
    reducing RAM usage for large datasets.
    
    Args:
        metadata_path: Path to the dataset metadata JSON file
        cache_size: Number of chunks to keep in cache (default: 2)
    """
    
    def __init__(self, metadata_path: str, cache_size: int = 2):
        self.metadata_path = Path(metadata_path)
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.dataset_name = self.metadata['dataset_name']
        self.total_samples = self.metadata['total_samples']
        self.num_chunks = self.metadata['num_chunks']
        self.chunks_info = self.metadata['chunks']
        
        # Build index: maps global sample index to (chunk_idx, local_idx)
        self.sample_index = []
        for chunk_info in self.chunks_info:
            chunk_idx = chunk_info['chunk_index']
            num_samples = chunk_info['num_samples']
            for local_idx in range(num_samples):
                self.sample_index.append((chunk_idx, local_idx))
        
        # Cache for loaded chunks
        self.cache_size = cache_size
        self.cache = {}  # {chunk_idx: (data, labels)}
        self.cache_order = []  # LRU tracking
        
        logger.info(f"Loaded ChunkedDataset: {self.dataset_name}")
        logger.info(f"Total samples: {self.total_samples}, Chunks: {self.num_chunks}")
    
    def _load_chunk(self, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a specific chunk from disk."""
        chunk_info = self.chunks_info[chunk_idx]
        
        data = np.load(chunk_info['data_path'])
        labels = np.load(chunk_info['label_path'])
        
        return data, labels
    
    def _get_chunk(self, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get chunk from cache or load from disk."""
        if chunk_idx in self.cache:
            # Move to end of LRU list
            self.cache_order.remove(chunk_idx)
            self.cache_order.append(chunk_idx)
            return self.cache[chunk_idx]
        
        # Load chunk
        data, labels = self._load_chunk(chunk_idx)
        
        # Add to cache
        self.cache[chunk_idx] = (data, labels)
        self.cache_order.append(chunk_idx)
        
        # Evict oldest chunk if cache is full
        if len(self.cache) > self.cache_size:
            oldest_chunk = self.cache_order.pop(0)
            del self.cache[oldest_chunk]
        
        return data, labels
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (data, label) as PyTorch tensors
        """
        chunk_idx, local_idx = self.sample_index[idx]
        data, labels = self._get_chunk(chunk_idx)
        
        return torch.FloatTensor(data[local_idx]), torch.FloatTensor(labels[local_idx])
    
    def get_info(self) -> Dict:
        """Get dataset information."""
        return {
            'name': self.dataset_name,
            'total_samples': self.total_samples,
            'num_chunks': self.num_chunks,
            'channels': self.metadata['channels'],
            'window_size': self.metadata['window_size']
        }


def create_dataloader(
    metadata_path: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    cache_size: int = 2
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a chunked dataset.
    
    Args:
        metadata_path: Path to dataset metadata JSON
        batch_size: Batch size for training
        shuffle: Whether to shuffle samples
        num_workers: Number of worker processes for data loading
        cache_size: Number of chunks to keep in cache per worker
    
    Returns:
        PyTorch DataLoader
    """
    dataset = ChunkedDataset(metadata_path, cache_size=cache_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# Convenience function for backward compatibility
def load_legacy_dataset_and_convert(
    legacy_data_path: str,
    legacy_label_path: str,
    output_base_path: str,
    dataset_name: str,
    max_chunk_size_gb: float = 4.0
) -> Path:
    """
    Convert a legacy single-file dataset to chunked format.
    
    Args:
        legacy_data_path: Path to existing .npy data file
        legacy_label_path: Path to existing .npy label file
        output_base_path: Directory for chunked output
        dataset_name: Name for the new chunked dataset
        max_chunk_size_gb: Maximum chunk size in GB
    
    Returns:
        Path to metadata file
    """
    logger.info(f"Converting legacy dataset to chunked format...")
    logger.info(f"Loading from: {legacy_data_path}")
    
    # Load legacy data
    data = np.load(legacy_data_path)
    labels = np.load(legacy_label_path)
    
    logger.info(f"Loaded data shape: {data.shape}, labels shape: {labels.shape}")
    
    # Create writer
    writer = ChunkedDatasetWriter(
        base_path=output_base_path,
        dataset_name=dataset_name,
        max_chunk_size_gb=max_chunk_size_gb,
        channels=data.shape[1] if len(data.shape) > 2 else 1,
        window_size=data.shape[2] if len(data.shape) > 2 else data.shape[1]
    )
    
    # Add samples
    for i in range(len(data)):
        writer.add_sample(data[i], labels[i])
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(data)} samples")
    
    # Finalize
    metadata_path = writer.finalize()
    
    logger.info(f"Conversion complete!")
    return metadata_path
