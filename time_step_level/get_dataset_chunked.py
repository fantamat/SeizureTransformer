"""
Memory-efficient dataset creation using chunked storage.

This version of get_dataset.py uses ChunkedDatasetWriter to avoid loading
all data into RAM at once. Data is written to disk in manageable chunks
as it's processed.
"""

from service.handle_data import get_file, get_data, get_dataloader, get_data_18
from utils.chunked_dataset import ChunkedDatasetWriter, _get_process_rss_mb
from tqdm import tqdm
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from sklearn.utils import shuffle as sklearn_shuffle
from scipy.signal import resample
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg


def parse_args():
    parser = argparse.ArgumentParser(description='Get ETHZ dataset with chunked storage')
    parser.add_argument('--window_size', type=int, default=15360, help='Window size')
    parser.add_argument('--alpha', type=float, default=0.7, help='Alpha value for training data')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta value for training data')
    parser.add_argument('--chunk_size_gb', type=float, default=4.0, help='Total write-buffer budget in GB shared across all concurrent writers')
    parser.add_argument('--data-siena', type=str, help='Use Siena dataset')
    parser.add_argument('--data-mit', type=str, help='Use MIT dataset')
    parser.add_argument('--data-ethz', type=str, help='Use ETHZ dataset')
    parser.add_argument('--output', type=str, default='./data/dataset/chunks', help='Output directory for chunked datasets')
    args = parser.parse_args()
    return args


def get_siena_chunked(window_size, output_dir, chunk_size_gb=4.0):
    """
    Process Siena dataset and save in chunked format.
    Returns metadata paths instead of data in memory.
    """
    print('=========Siena (Chunked)=========')
    
    # Create chunked writers for each category
    writers = {
        'full': ChunkedDatasetWriter(
            output_dir,
            f'siena_18_full_{window_size}',
            max_chunk_size_gb=chunk_size_gb,
            channels=18,
            window_size=window_size
        ),
        'no': ChunkedDatasetWriter(
            output_dir,
            f'siena_18_no_{window_size}',
            max_chunk_size_gb=chunk_size_gb,
            channels=18,
            window_size=window_size
        ),
        'valid': ChunkedDatasetWriter(
            output_dir,
            f'siena_18_valid_{window_size}',
            max_chunk_size_gb=chunk_size_gb,
            channels=18,
            window_size=window_size
        )
    }
    
    counts = {'full': 0, 'no': 0, 'valid': 0}

    data_path = './data/BIDS_Siena'
    if args.data_siena:
        data_path = args.data_siena
    file_list, label_list = get_file(data_path)

    progress = tqdm(file_list, desc='Processing Siena')
    for edf_path in progress:
        eeg = Eeg.loadEdfAutoDetectMontage(edf_path)
        eeg.reReferenceToBipolar()
        data = eeg.data
        # normalize data
        data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        # resample data
        sample_rate = eeg.fs
        if sample_rate != 256:
            new_n_samples = int(data.shape[1] / sample_rate * float(256))
            data = resample(data, new_n_samples, axis=1)
        if data.shape[1] < window_size:
            continue
        name = edf_path.split('/')[-1].split('.')[0]
        label_name = name.split('_')
        label_name = '_'.join(label_name[:-1]) + '_events.tsv'
        label_path = os.path.join('/'.join(edf_path.split('/')[:-1]), label_name)
        ref = Annotations.loadTsv(label_path)
        if ref.events:
            detect_label = ref.getMask(fs=256)
        else:
            detect_label = np.zeros(data.shape[1], dtype=int)

        dataloader = get_dataloader(data, detect_label, window_size=window_size)
        for batch in dataloader:
            X = batch['X']
            y_detect = batch['y_detect']

            X = X.detach().numpy()
            y_detect = y_detect.detach().numpy()

            for mini_batch in range(X.shape[0]):
                seizure_sample_cnt = np.sum(y_detect[mini_batch, :])
                temp = X[mini_batch]
                
                if seizure_sample_cnt == 0:
                    writers['no'].add_sample(temp, y_detect[mini_batch, :])
                    counts['no'] += 1
                elif seizure_sample_cnt == window_size:
                    writers['full'].add_sample(temp, y_detect[mini_batch, :])
                    counts['full'] += 1
                else:
                    writers['valid'].add_sample(temp, y_detect[mini_batch, :])
                    counts['valid'] += 1
                    
            progress.set_postfix({
                'valid': counts['valid'],
                'full': counts['full'],
                'no': counts['no']
            })
    
    print(f"Full windows: {counts['full']}")
    print(f"No seizure windows: {counts['no']}")
    print(f"Valid windows: {counts['valid']}")
    
    # Finalize all writers
    metadata_paths = {
        'full': writers['full'].finalize(),
        'no': writers['no'].finalize(),
        'valid': writers['valid'].finalize()
    }
    
    return metadata_paths, counts


def get_mit_chunked(window_size, output_dir, chunk_size_gb=4.0):
    """
    Process MIT dataset and save in chunked format.
    """
    print('=========MIT (Chunked)=========')
    
    writers = {
        'full': ChunkedDatasetWriter(
            output_dir,
            f'mit_18_full_{window_size}',
            max_chunk_size_gb=chunk_size_gb,
            channels=18,
            window_size=window_size
        ),
        'no': ChunkedDatasetWriter(
            output_dir,
            f'mit_18_no_{window_size}',
            max_chunk_size_gb=chunk_size_gb,
            channels=18,
            window_size=window_size
        ),
        'valid': ChunkedDatasetWriter(
            output_dir,
            f'mit_18_valid_{window_size}',
            max_chunk_size_gb=chunk_size_gb,
            channels=18,
            window_size=window_size
        )
    }
    
    counts = {'full': 0, 'no': 0, 'valid': 0}

    data_path = './data/BIDS_CHB-MIT'
    if args.data_mit:
        data_path = args.data_mit
    file_list, label_list = get_file(data_path)

    progress = tqdm(file_list, desc='Processing MIT')
    for edf_path in progress:
        eeg = Eeg.loadEdfAutoDetectMontage(edf_path)
        data = eeg.data
        # normalize data
        data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        # resample data
        sample_rate = eeg.fs
        if sample_rate != 256:
            new_n_samples = int(data.shape[1] / sample_rate * float(256))
            data = resample(data, new_n_samples, axis=1)
        if data.shape[1] < window_size:
            continue
        name = edf_path.split('/')[-1].split('.')[0]
        label_name = name.split('_')
        label_name = '_'.join(label_name[:-1]) + '_events.tsv'
        label_path = os.path.join('/'.join(edf_path.split('/')[:-1]), label_name)
        ref = Annotations.loadTsv(label_path)
        detect_label = ref.getMask(fs=256)

        dataloader = get_dataloader(data, detect_label, window_size=window_size)
        for batch in dataloader:
            X = batch['X']
            y_detect = batch['y_detect']

            X = X.detach().numpy()
            y_detect = y_detect.detach().numpy()

            for mini_batch in range(X.shape[0]):
                seizure_sample_cnt = np.sum(y_detect[mini_batch, :])
                temp = X[mini_batch]
                
                if seizure_sample_cnt == 0:
                    writers['no'].add_sample(temp, y_detect[mini_batch, :])
                    counts['no'] += 1
                elif seizure_sample_cnt == window_size:
                    writers['full'].add_sample(temp, y_detect[mini_batch, :])
                    counts['full'] += 1
                else:
                    writers['valid'].add_sample(temp, y_detect[mini_batch, :])
                    counts['valid'] += 1
                    
            progress.set_postfix({
                'valid': counts['valid'],
                'full': counts['full'],
                'no': counts['no']
            })
    
    print(f"Full windows: {counts['full']}")
    print(f"No seizure windows: {counts['no']}")
    print(f"Valid windows: {counts['valid']}")
    
    metadata_paths = {
        'full': writers['full'].finalize(),
        'no': writers['no'].finalize(),
        'valid': writers['valid'].finalize()
    }
    
    return metadata_paths, counts


def get_ethz_chunked(window_size, output_dir, chunk_size_gb=4.0):
    """
    Process ETHZ dataset and save in chunked format.
    """
    print('=========ETHZ (Chunked)=========')
    
    writers = {
        'full': ChunkedDatasetWriter(
            output_dir,
            f'ethz_18_full_{window_size}',
            max_chunk_size_gb=chunk_size_gb,
            channels=18,
            window_size=window_size
        ),
        'no': ChunkedDatasetWriter(
            output_dir,
            f'ethz_18_no_{window_size}',
            max_chunk_size_gb=chunk_size_gb,
            channels=18,
            window_size=window_size
        ),
        'valid': ChunkedDatasetWriter(
            output_dir,
            f'ethz_18_valid_{window_size}',
            max_chunk_size_gb=chunk_size_gb,
            channels=18,
            window_size=window_size
        )
    }
    
    counts = {'full': 0, 'no': 0, 'valid': 0}

    data_path = './data/v2.0.3/edf/train'
    if args.data_ethz:
        data_path = args.data_ethz

    file_list, label_list = get_file(data_path)

    progress = tqdm(range(len(file_list)), desc='Processing ETHZ')
    for i in progress:
        data_file, label_file = file_list[i], label_list[i]
        try:
            data, detect_label = get_data_18(data_file, label_file)
        except:
            continue
        if data.shape[1] < window_size:
            continue

        logger.info(
            f"[ETHZ file {i}/{len(file_list)}] data={data.nbytes / (1024**2):.1f} MB "
            f"shape={data.shape}  detect_label={detect_label.nbytes / 1024:.1f} kB  "
            f"process_rss={_get_process_rss_mb():.0f} MB"
        )

        dataloader = get_dataloader(data, detect_label, window_size=window_size)
        for batch in dataloader:
            X = batch['X']
            y_detect = batch['y_detect']

            X = X.detach().numpy()
            y_detect = y_detect.detach().numpy()

            logger.debug(
                f"  batch X={X.nbytes / 1024:.1f} kB shape={X.shape}  "
                f"y_detect={y_detect.nbytes / 1024:.1f} kB"
            )

            for mini_batch in range(X.shape[0]):
                seizure_sample_cnt = np.sum(y_detect[mini_batch, :])
                temp = X[mini_batch]
                
                if seizure_sample_cnt == 0:
                    writers['no'].add_sample(temp, y_detect[mini_batch, :])
                    counts['no'] += 1
                elif seizure_sample_cnt == window_size:
                    writers['full'].add_sample(temp, y_detect[mini_batch, :])
                    counts['full'] += 1
                else:
                    writers['valid'].add_sample(temp, y_detect[mini_batch, :])
                    counts['valid'] += 1
                    
            progress.set_postfix({
                'valid': counts['valid'],
                'full': counts['full'],
                'no': counts['no']
            })
    
    print(f"Full windows: {counts['full']}")
    print(f"No seizure windows: {counts['no']}")
    print(f"Valid windows: {counts['valid']}")
    
    metadata_paths = {
        'full': writers['full'].finalize(),
        'no': writers['no'].finalize(),
        'valid': writers['valid'].finalize()
    }
    
    return metadata_paths, counts


def create_combined_training_dataset(
    output_dir,
    all_metadata_paths,
    all_counts,
    alpha,
    beta,
    window_size,
    chunk_size_gb=4.0,
    seed=42
):
    """
    Create the final combined training dataset with proper class balancing.
    
    This function loads chunks from different categories and combines them
    into a final training dataset with appropriate ratios.
    
    Args:
        all_metadata_paths: Dict of metadata paths from all datasets
        all_counts: Dict of sample counts from all datasets
        alpha: Ratio of full seizure windows to valid windows
        beta: Ratio of no-seizure windows to valid windows
        window_size: Window size
        chunk_size_gb: Maximum chunk size for output
        seed: Random seed for shuffling
    """
    print('='*10, 'Creating Combined Training Dataset', '='*10)
    
    # Calculate target counts
    total_valid = sum(counts['valid'] for counts in all_counts.values())
    target_full = int(total_valid * alpha)
    target_no = int(total_valid * beta)
    
    print(f"Target samples - Valid: {total_valid}, Full: {target_full}, No: {target_no}")
    
    # Create final writer
    final_writer = ChunkedDatasetWriter(
        output_dir,
        f'full_train_{alpha}_{beta}_{window_size}',
        max_chunk_size_gb=chunk_size_gb,
        channels=18,
        window_size=window_size
    )
    
    from time_step_level.utils.chunked_dataset import ChunkedDataset
    
    # Helper function to add samples from a dataset with limit
    def add_samples_from_dataset(metadata_path, max_samples=None):
        dataset = ChunkedDataset(metadata_path, cache_size=1)
        total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        
        # Create shuffled indices
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))[:total]
        
        for idx in tqdm(indices, desc=f"Loading {metadata_path.stem}"):
            data, label = dataset[idx]
            final_writer.add_sample(data.numpy(), label.numpy())
    
    # Add valid samples (all of them)
    print("\nAdding valid samples...")
    for dataset_name, metadata_paths in all_metadata_paths.items():
        add_samples_from_dataset(metadata_paths['valid'])
    
    # Add full samples (limited by alpha)
    print(f"\nAdding full seizure samples (max {target_full})...")
    added_full = 0
    for dataset_name, metadata_paths in all_metadata_paths.items():
        remaining = target_full - added_full
        if remaining <= 0:
            break
        dataset = ChunkedDataset(metadata_paths['full'], cache_size=1)
        to_add = min(len(dataset), remaining)
        
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))[:to_add]
        
        for idx in tqdm(indices, desc=f"Loading {dataset_name} full"):
            data, label = dataset[idx]
            final_writer.add_sample(data.numpy(), label.numpy())
            added_full += 1
    
    # Add no-seizure samples (limited by beta)
    print(f"\nAdding no-seizure samples (max {target_no})...")
    added_no = 0
    for dataset_name, metadata_paths in all_metadata_paths.items():
        remaining = target_no - added_no
        if remaining <= 0:
            break
        dataset = ChunkedDataset(metadata_paths['no'], cache_size=1)
        to_add = min(len(dataset), remaining)
        
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))[:to_add]
        
        for idx in tqdm(indices, desc=f"Loading {dataset_name} no"):
            data, label = dataset[idx]
            final_writer.add_sample(data.numpy(), label.numpy())
            added_no += 1
    
    # Finalize
    print("\nFinalizing combined dataset...")
    metadata_path = final_writer.finalize()
    
    print(f"Combined dataset created: {metadata_path}")
    print(f"Total samples - Valid: {total_valid}, Full: {added_full}, No: {added_no}")
    print(f"Total: {total_valid + added_full + added_no}")
    
    return metadata_path


if __name__ == '__main__':
    args = parse_args()
    window_size = args.window_size
    alpha = args.alpha
    beta = args.beta
    chunk_size_gb = args.chunk_size_gb

    all_metadata_paths = {}
    all_counts = {}

    # Process Siena
    if args.data_siena:
        metadata_paths, counts = get_siena_chunked(window_size, args.output, chunk_size_gb)
        all_metadata_paths['siena'] = metadata_paths
        all_counts['siena'] = counts

    # Process ETHZ
    if args.data_ethz:
        metadata_paths, counts = get_ethz_chunked(window_size, args.output, chunk_size_gb)
        all_metadata_paths['ethz'] = metadata_paths
        all_counts['ethz'] = counts

    # Process MIT
    if args.data_mit:
        metadata_paths, counts = get_mit_chunked(window_size, args.output, chunk_size_gb)
        all_metadata_paths['mit'] = metadata_paths
        all_counts['mit'] = counts

    # Create combined training dataset
    final_metadata_path = create_combined_training_dataset(
        args.output,
        all_metadata_paths,
        all_counts,
        alpha,
        beta,
        window_size,
        chunk_size_gb
    )
    
    print(f"\n{'='*50}")
    print(f"Dataset creation complete!")
    print(f"Training metadata: {final_metadata_path}")
    print(f"{'='*50}")
