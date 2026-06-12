"""
Script to convert raw EEG datasets to BIDS (Brain Imaging Data Structure) format.

This script converts EEG data from Siena, CHB-MIT, and TUH-EEG (ETHZ) datasets
into BIDS structure manually (without mne-bids), including standard sidecar files
and dataset metadata.

By default, EDF files are symlinked (not copied) to save disk space. Use --copy
to create actual copies if needed.

Requirements:
    pip install mne

Usage:
    # Convert Siena dataset (creates symlinks by default)
    python convert_to_bids.py --dataset siena \
        --input /home/fanta/data/eeg/Siena/ \
        --output ./data/BIDS_Siena
    
    # Convert CHB-MIT dataset with actual file copies
    python convert_to_bids.py --dataset chb-mit \
        --input /home/fanta/data/eeg/chb-mit-scalp-eeg-database-1.0.0/ \
        --output ./data/BIDS_CHB-MIT \
        --copy
    
    # Convert TUH-EEG (ETHZ) dataset
    python convert_to_bids.py --dataset tuh-eeg \
        --input /home/fanta/data/eeg/nedc/data/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf/train/ \
        --output ./data/BIDS_TUH-EEG
    
    # Dry run to preview conversion
    python convert_to_bids.py --dataset siena \
        --input /path/to/Siena/ \
        --output ./data/BIDS_Siena \
        --dry-run
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import re

try:
    import mne
except ImportError:
    print("ERROR: mne is required but not installed.")
    print("Please install it with: pip install mne")
    sys.exit(1)

from time_step_level.utils.convert_to_bids_utils import ManualBIDSPath, ManualBIDSWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress MNE's verbose output
mne.set_log_level('WARNING')


class BIDSConverter:
    """Base class for manual BIDS conversion."""
    
    def __init__(self, input_path: Path, output_path: Path, dry_run: bool = False, copy_files: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.dry_run = dry_run
        self.copy_files = copy_files
        self.converted_files = 0
        self.skipped_files = 0
        self.failed_files = 0
        self.bids_writer = ManualBIDSWriter(
            root=self.output_path,
            dry_run=self.dry_run,
            copy_files=self.copy_files,
        )
        
    def sanitize_label(self, label: str) -> str:
        """Sanitize label to BIDS format (alphanumeric only)."""
        # Remove any non-alphanumeric characters (including underscores, hyphens, slashes)
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', label)
        return sanitized if sanitized else 'unknown'
    
    def clean_channel_names(self, raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, bool]:
        """Clean electrode/channel names by removing common prefixes like 'EEG '.
        
        This ensures compatibility with epilepsy2bids library which expects 
        standard 10-20 electrode names (e.g., 'Fp1' not 'EEG Fp1').
        
        Args:
            raw: MNE Raw object with potentially prefixed channel names
            
        Returns:
            MNE Raw object with cleaned channel names
        """
        old_names = raw.ch_names
        new_names = []
        
        for name in old_names:
            # Remove common prefixes
            cleaned = re.sub(r'^EEG\s+', '', name, flags=re.IGNORECASE)
            cleaned = re.sub(r'^ECG\s+', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^EOG\s+', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^EMG\s+', '', cleaned, flags=re.IGNORECASE)
            new_names.append(cleaned)
        
        # Rename channels if any changes were made
        if old_names != new_names:
            mapping = {old: new for old, new in zip(old_names, new_names)}
            raw.rename_channels(mapping)
            logger.debug(f"Cleaned {len([o for o, n in zip(old_names, new_names) if o != n])} channel names")
        
        return raw, old_names != new_names
    


    def get_sanitized_edf_file(self, edf_path: Path) -> Tuple[mne.io.BaseRaw, bool]:
        """Load EDF and clean channels, returning whether changes were made."""

        # Read EDF with MNE - load into memory to allow channel modifications
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
        raw, changed = self.clean_channel_names(raw)
        return raw, changed
    

    def create_dataset_description(self, dataset_name: str, authors: Optional[List[str]] = None):
        """Create BIDS root metadata files."""
        self.bids_writer.create_dataset_description(dataset_name=dataset_name, authors=authors)
        self.bids_writer.write_participants_files()
        self.bids_writer.write_readme(dataset_name=dataset_name, source_description=str(self.input_path))
        logger.info(f"Created dataset_description.json")
    
    def write_raw_to_bids(
        self, 
        raw: mne.io.BaseRaw, 
        subject: str, 
        session: str, 
        source_edf_path: Path,
        task: str = "seizure",
        run: Optional[str] = None,
        channels_cleaned: bool = False,
    ) -> Optional[ManualBIDSPath]:
        """
        Write MNE Raw object to BIDS format manually.
        
        Args:
            raw: MNE Raw object with data
            subject: Subject/participant ID
            session: Session ID  
            task: Task name (default: seizure)
            run: Run number (optional)
            
        Returns:
            ManualBIDSPath object if successful, None otherwise
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would write: sub-{subject}, ses-{session}, task-{task}")
            return self.bids_writer.make_bids_path(subject=subject, session=session, task=task, run=run)
        
        try:
            bids_path = self.bids_writer.write_raw_eeg(
                raw=raw,
                source_edf_path=source_edf_path,
                subject=subject,
                session=session,
                task=task,
                run=run,
                channels_cleaned=channels_cleaned,
            )
            return bids_path
            
        except Exception as e:
            logger.error(f"Failed to write BIDS: {e}")
            return None
    
    def convert(self):
        """Override in subclass."""
        raise NotImplementedError()


class SienaConverter(BIDSConverter):
    """Convert Siena dataset to BIDS format."""
    
    def find_edf_files(self) -> List[Tuple[Path, Path]]:
        """Find EDF files and their corresponding annotation files."""
        edf_files = []
        for root, dirs, files in os.walk(self.input_path):
            for file in files:
                if file.endswith('.edf') and not file.startswith('_'):
                    edf_path = Path(root) / file
                    base_name = file.replace('.edf', '')
                    
                    # Look for annotation files
                    annotation_patterns = [
                        f"{base_name}.txt",
                        f"{base_name}_annotations.txt",
                        f"{base_name}.csv",
                        f"{base_name}.tsv",
                    ]
                    
                    annotation_path = None
                    for pattern in annotation_patterns:
                        candidate = Path(root) / pattern
                        if candidate.exists():
                            annotation_path = candidate
                            break
                    
                    edf_files.append((edf_path, annotation_path))
        
        return edf_files
    
    def parse_siena_annotations(self, annotation_file: Path) -> List[Tuple[float, float, str]]:
        """Parse Siena annotations and return as list of (onset, duration, description) tuples."""
        annotations = []
        
        if not annotation_file or not annotation_file.exists():
            return annotations
        
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = re.split(r'[,\t]+', line)
            
            if len(parts) >= 2:
                try:
                    start = float(parts[0])
                    # Check if second value is end time or duration
                    value = float(parts[1])
                    if value > start and value > 100:  # Likely end time
                        duration = value - start
                    else:  # Likely duration
                        duration = value
                    
                    description = parts[2] if len(parts) > 2 else 'seizure'
                    annotations.append((start, duration, description))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse line: {line} - {e}")
        
        return annotations
    
    def convert(self):
        """Convert Siena dataset to BIDS."""
        logger.info("Converting Siena dataset to BIDS format...")
        
        edf_files = self.find_edf_files()
        logger.info(f"Found {len(edf_files)} EDF files")
        
        if not edf_files:
            logger.error("No EDF files found in input directory")
            return
        
        for edf_path, annotation_path in edf_files:
            try:
                # Extract subject and session from filename
                filename = edf_path.stem
                match = re.search(r'PN(\d+)[-_]?(\w+)?', filename, re.IGNORECASE)
                if match:
                    subject = f"PN{match.group(1)}"
                    session = self.sanitize_label(match.group(2)) if match.group(2) else "01"
                else:
                    parts = filename.split('_')
                    subject = self.sanitize_label(parts[0])
                    session = self.sanitize_label(parts[1]) if len(parts) > 1 else "01"
                
                raw, channels_cleaned = self.get_sanitized_edf_file(edf_path)
                
                # Add annotations if available
                if annotation_path:
                    annot_data = self.parse_siena_annotations(annotation_path)
                    if annot_data:
                        onset, duration, description = zip(*annot_data)
                        annotations = mne.Annotations(
                            onset=onset,
                            duration=duration,
                            description=description
                        )
                        raw.set_annotations(annotations)
                        logger.debug(f"Added {len(annot_data)} annotations")
                
                # Write to BIDS
                bids_path = self.write_raw_to_bids(
                    raw,
                    subject,
                    session,
                    source_edf_path=edf_path,
                    channels_cleaned=channels_cleaned,
                )
                
                if bids_path:
                    logger.info(f"Converted {edf_path.name} -> {bids_path.basename}")
                    self.converted_files += 1
                else:
                    self.failed_files += 1
                
                
            except Exception as e:
                logger.error(f"Failed to convert {edf_path}: {e}")
                self.failed_files += 1
        
        # Create dataset description
        self.create_dataset_description("Siena Scalp EEG Database")


class CHBMITConverter(BIDSConverter):
    """Convert CHB-MIT Scalp EEG Database to BIDS format."""
    
    def parse_chb_summary(self, summary_file: Path, edf_file: str) -> List[Tuple[float, float, str]]:
        """Parse CHB-MIT summary file for seizure annotations."""
        annotations = []
        
        if not summary_file.exists():
            return annotations
        
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        
        current_file = None
        in_file_section = False
        seizure_data = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('File Name:'):
                current_file = line.split(':')[1].strip()
                in_file_section = (current_file == edf_file)
                seizure_data = {}
                
            elif in_file_section:
                if 'Start Time:' in line:
                    match = re.search(r'(\d+)\s*sec', line)
                    if match:
                        onset = int(match.group(1))
                        seizure_data['onset'] = onset
                elif 'End Time:' in line:
                    match = re.search(r'(\d+)\s*sec', line)
                    if match and 'onset' in seizure_data:
                        end = int(match.group(1))
                        duration = end - seizure_data['onset']
                        annotations.append((float(seizure_data['onset']), float(duration), 'seizure'))
                        seizure_data = {}
        
        return annotations
    
    def convert(self):
        """Convert CHB-MIT dataset to BIDS."""
        logger.info("Converting CHB-MIT dataset to BIDS format...")
        
        # CHB-MIT is organized by patient folders (chb01, chb02, etc.)
        for patient_dir in sorted(self.input_path.iterdir()):
            if not patient_dir.is_dir() or not patient_dir.name.startswith('chb'):
                continue
            
            subject = patient_dir.name.replace('chb', 'CHB')
            logger.info(f"Processing participant: {subject}")
            
            # Look for summary file
            summary_file = patient_dir / f"{patient_dir.name}-summary.txt"
            
            # Process each EDF file
            edf_files = sorted(patient_dir.glob('*.edf'))
            
            for idx, edf_path in enumerate(edf_files, 1):
                if edf_path.name.startswith('_'):
                    continue
                
                try:
                    session = f"{idx:03d}"
                    
                    # Read EDF with MNE and clean channel names
                    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
                    raw, channels_cleaned = self.clean_channel_names(raw)
                    
                    # Parse and add annotations from summary file
                    annot_data = self.parse_chb_summary(summary_file, edf_path.name)
                    if annot_data:
                        onset, duration, description = zip(*annot_data)
                        annotations = mne.Annotations(
                            onset=onset,
                            duration=duration,
                            description=description
                        )
                        raw.set_annotations(annotations)
                    
                    # Write to BIDS
                    bids_path = self.write_raw_to_bids(
                        raw,
                        subject,
                        session,
                        source_edf_path=edf_path,
                        channels_cleaned=channels_cleaned,
                    )
                    
                    if bids_path:
                        logger.info(f"Converted {edf_path.name} ({len(annot_data)} seizure events)")
                        self.converted_files += 1
                    else:
                        self.failed_files += 1
                    
                    
                except Exception as e:
                    logger.error(f"Failed to convert {edf_path}: {e}")
                    self.failed_files += 1
        
        # Create dataset description
        self.create_dataset_description("CHB-MIT Scalp EEG Database")


class TUHEEGConverter(BIDSConverter):
    """Convert TUH-EEG Seizure Corpus to BIDS format."""
    
    def find_edf_files(self) -> List[Tuple[Path, Path]]:
        """Find EDF files and corresponding annotation files."""
        edf_files = []
        
        for root, dirs, files in os.walk(self.input_path):
            for file in files:
                if file.endswith('.edf') and not file.startswith('_'):
                    edf_path = Path(root) / file
                    base_name = file.replace('.edf', '')
                    
                    # Look for .tse, .csv_bi, or .lbl files
                    annotation_file = None
                    for ext in ['.tse', '.csv_bi', '.lbl']:
                        candidate = Path(root) / f"{base_name}{ext}"
                        if candidate.exists():
                            annotation_file = candidate
                            break
                    
                    edf_files.append((edf_path, annotation_file))
        
        return edf_files
    
    def parse_tuh_annotations(self, annotation_file: Path) -> List[Tuple[float, float, str]]:
        """Parse TUH-EEG annotation files."""
        annotations = []
        
        if not annotation_file or not annotation_file.exists():
            return annotations
        
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('version'):
                continue
            
            parts = re.split(r'[,\s]+', line)
            
            if len(parts) >= 3:
                try:
                    start = float(parts[0])
                    stop = float(parts[1])
                    label = parts[2].lower()
                    
                    # Include seizure events and other relevant events
                    if label != 'bckg' and label != 'background':
                        annotations.append((start, stop - start, label))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse line: {line} - {e}")
        
        return annotations
    
    def convert(self):
        """Convert TUH-EEG dataset to BIDS."""
        logger.info("Converting TUH-EEG dataset to BIDS format...")
        
        edf_files = self.find_edf_files()
        logger.info(f"Found {len(edf_files)} EDF files")
        
        if not edf_files:
            logger.error("No EDF files found in input directory")
            return
        
        # Track sessions per participant
        participant_sessions = {}
        
        for edf_path, annotation_path in edf_files:
            try:
                # Extract participant from path structure
                path_parts = edf_path.parts
                participant = None
                
                # TUH structure: .../patient_id/session_id/file.edf
                for i in range(len(path_parts) - 1, 0, -1):
                    if path_parts[i].endswith('.edf'):
                        continue
                    if participant is None:
                        # Use 2 levels up from the file
                        for j in range(max(0, i-2), i):
                            if path_parts[j] not in ['train', 'dev', 'eval']:
                                participant = self.sanitize_label(path_parts[j])
                                break
                        break
                
                if not participant:
                    participant = self.sanitize_label(edf_path.stem)
                
                # Track session number for this participant
                if participant not in participant_sessions:
                    participant_sessions[participant] = 0
                participant_sessions[participant] += 1
                session = f"{participant_sessions[participant]:03d}"
                
                # Read EDF with MNE and clean channel names
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
                raw, channels_cleaned = self.clean_channel_names(raw)
                
                # Add annotations if available
                annot_data = []
                if annotation_path:
                    annot_data = self.parse_tuh_annotations(annotation_path)
                    if annot_data:
                        onset, duration, description = zip(*annot_data)
                        annotations = mne.Annotations(
                            onset=onset,
                            duration=duration,
                            description=description
                        )
                        raw.set_annotations(annotations)
                
                # Write to BIDS
                bids_path = self.write_raw_to_bids(
                    raw,
                    participant,
                    session,
                    source_edf_path=edf_path,
                    channels_cleaned=channels_cleaned,
                )
                
                if bids_path:
                    num_events = len(annot_data)
                    logger.info(f"Converted {edf_path.name} ({num_events} events)")
                    self.converted_files += 1
                else:
                    self.failed_files += 1
                
            except Exception as e:
                logger.error(f"Failed to convert {edf_path}: {e}")
                self.failed_files += 1
        
        # Create dataset description
        self.create_dataset_description("TUH EEG Seizure Corpus")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert EEG datasets to BIDS format (manual writer, no mne-bids)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['siena', 'chb-mit', 'tuh-eeg'],
        help='Dataset type to convert'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing raw dataset'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for BIDS format'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be converted without actually converting'
    )
    
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy EDF files instead of creating symbolic links (default: use symlinks to save space)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return 1
    
    if not args.dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Select converter based on dataset type
    converters = {
        'siena': SienaConverter,
        'chb-mit': CHBMITConverter,
        'tuh-eeg': TUHEEGConverter
    }
    
    converter_class = converters[args.dataset]
    converter = converter_class(input_path, output_path, dry_run=args.dry_run, copy_files=args.copy)
    
    # Run conversion
    logger.info("=" * 80)
    logger.info(f"BIDS Conversion Configuration")
    logger.info("=" * 80)
    logger.info(f"Dataset type:     {args.dataset}")
    logger.info(f"Input directory:  {input_path.resolve()}")
    logger.info(f"Output directory: {output_path.resolve()}")
    logger.info(f"File handling:    {'Copy' if args.copy else 'Symlink (saves disk space)'}")
    logger.info(f"Dry run:          {args.dry_run}")
    logger.info(f"Using:            Manual BIDS writer")
    logger.info("=" * 80)
    
    converter.convert()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Conversion Summary")
    logger.info("=" * 80)
    logger.info(f"✓ Successfully converted: {converter.converted_files} files")
    if converter.skipped_files > 0:
        logger.info(f"⊘ Skipped:                {converter.skipped_files} files")
    if converter.failed_files > 0:
        logger.info(f"✗ Failed:                 {converter.failed_files} files")
    logger.info("=" * 80)
    
    if args.dry_run:
        logger.info("\nThis was a dry run. No files were actually converted.")
        logger.info(f"Run without --dry-run to perform the conversion.")
    
    return 0 if converter.failed_files == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
