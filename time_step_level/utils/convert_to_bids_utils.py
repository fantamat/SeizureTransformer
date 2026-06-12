"""Utility helpers for manual EEG BIDS writing.

This module intentionally avoids mne-bids so conversion can run with only MNE.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import mne


def sanitize_bids_label(label: str) -> str:
	"""Return a BIDS-safe label (alphanumeric only)."""
	cleaned = "".join(ch for ch in str(label) if ch.isalnum())
	return cleaned if cleaned else "unknown"


@dataclass
class ManualBIDSPath:
	"""Small replacement for mne-bids BIDSPath used by converter logging."""

	root: Path
	subject: str
	session: str
	task: str
	run: Optional[str] = None
	datatype: str = "eeg"
	extension: str = ".edf"

	@property
	def basename(self) -> str:
		entities = [
			f"sub-{self.subject}",
			f"ses-{self.session}",
			f"task-{self.task}",
		]
		if self.run:
			entities.append(f"run-{self.run}")
		entities.append("eeg")
		return "_".join(entities) + self.extension

	@property
	def directory(self) -> Path:
		return self.root / f"sub-{self.subject}" / f"ses-{self.session}" / self.datatype

	@property
	def fpath(self) -> Path:
		return self.directory / self.basename


class ManualBIDSWriter:
	"""Write BIDS EEG files and sidecars manually."""

	def __init__(self, root: Path, dry_run: bool = False, copy_files: bool = False):
		self.root = root
		self.dry_run = dry_run
		self.copy_files = copy_files
		self._participants: Dict[str, Dict[str, str]] = {}

	def make_bids_path(
		self,
		subject: str,
		session: str,
		task: str = "seizure",
		run: Optional[str] = None,
	) -> ManualBIDSPath:
		return ManualBIDSPath(
			root=self.root,
			subject=sanitize_bids_label(subject),
			session=sanitize_bids_label(session),
			task=sanitize_bids_label(task),
			run=sanitize_bids_label(run) if run else None,
		)

	def create_dataset_description(
		self,
		dataset_name: str,
		authors: Optional[List[str]] = None,
		data_license: str = "Unknown",
	) -> None:
		if self.dry_run:
			return

		self.root.mkdir(parents=True, exist_ok=True)
		if authors is None:
			authors = ["SeizureTransformer Conversion"]

		description = {
			"Name": dataset_name,
			"BIDSVersion": "1.8.0",
			"DatasetType": "raw",
			"License": data_license,
			"Authors": authors,
			"GeneratedBy": [
				{
					"Name": "convert_to_bids.py",
					"Version": "manual",
					"Description": "Manual BIDS conversion without mne-bids",
				}
			],
			"HowToAcknowledge": "Please cite the source dataset and this conversion pipeline.",
		}

		with open(self.root / "dataset_description.json", "w", encoding="utf-8") as f:
			json.dump(description, f, indent=2)

	def register_participant(self, subject: str, raw: mne.io.BaseRaw) -> None:
		sid = sanitize_bids_label(subject)
		if sid in self._participants:
			return

		subject_info = raw.info.get("subject_info") or {}
		sex_map = {0: "n/a", 1: "M", 2: "F"}
		sex_value: Union[int, str] = subject_info.get("sex", 0)
		sex = sex_map.get(int(sex_value), "n/a") if isinstance(sex_value, int) else "n/a"
		age = subject_info.get("age", "n/a")

		self._participants[sid] = {
			"participant_id": f"sub-{sid}",
			"sex": str(sex),
			"age": str(age),
		}

	def write_participants_files(self) -> None:
		if self.dry_run:
			return

		participants_tsv = self.root / "participants.tsv"
		participants_json = self.root / "participants.json"

		rows = [self._participants[sid] for sid in sorted(self._participants)]
		with open(participants_tsv, "w", newline="", encoding="utf-8") as f:
			writer = csv.DictWriter(f, fieldnames=["participant_id", "sex", "age"], delimiter="\t")
			writer.writeheader()
			writer.writerows(rows)

		meta = {
			"participant_id": {"Description": "Unique participant identifier."},
			"sex": {
				"Description": "Biological sex where available.",
				"Levels": {"M": "male", "F": "female", "n/a": "not available"},
			},
			"age": {"Description": "Age in years where available.", "Units": "years"},
		}
		with open(participants_json, "w", encoding="utf-8") as f:
			json.dump(meta, f, indent=2)

	def write_raw_eeg(
		self,
		raw: mne.io.BaseRaw,
		source_edf_path: Path,
		subject: str,
		session: str,
		task: str = "seizure",
		run: Optional[str] = None,
		channels_cleaned: bool = False,
	) -> ManualBIDSPath:
		bids_path = self.make_bids_path(subject=subject, session=session, task=task, run=run)
		self.register_participant(subject, raw)

		if self.dry_run:
			return bids_path

		bids_path.directory.mkdir(parents=True, exist_ok=True)
		target_edf = bids_path.fpath
		self._replace_file_if_exists(target_edf)

		if channels_cleaned:
			raw.export(target_edf, fmt="edf", overwrite=True)
		elif self.copy_files:
			shutil.copy2(source_edf_path, target_edf)
		else:
			os.symlink(source_edf_path.resolve(), target_edf)

		self._write_eeg_json(raw, bids_path)
		self._write_channels_tsv(raw, bids_path)
		self._write_events_files(raw, bids_path)

		return bids_path

	def _replace_file_if_exists(self, path: Path) -> None:
		if path.is_symlink() or path.exists():
			path.unlink()

	def _sidecar_path(self, bids_path: ManualBIDSPath, suffix: str, include_datatype_suffix: bool = False) -> Path:
		recording_stem = bids_path.basename.rsplit(".", 1)[0]
		entity_stem = recording_stem[:-4] if recording_stem.endswith("_eeg") else recording_stem
		base_stem = recording_stem if include_datatype_suffix else entity_stem
		return bids_path.directory / f"{base_stem}_{suffix}"

	def _write_eeg_json(self, raw: mne.io.BaseRaw, bids_path: ManualBIDSPath) -> None:
		eeg_json_path = self._sidecar_path(bids_path, "json", include_datatype_suffix=True)
		line_freq = raw.info.get("line_freq")
		sfreq = float(raw.info["sfreq"])
		duration = float(raw.n_times / sfreq)

		device_info = raw.info.get("device_info") or {}
		manufacturer = device_info.get("type") or "n/a"

		channel_types = raw.get_channel_types()
		eeg_count = sum(1 for t in channel_types if t == "eeg")
		eog_count = sum(1 for t in channel_types if t == "eog")
		ecg_count = sum(1 for t in channel_types if t == "ecg")
		emg_count = sum(1 for t in channel_types if t == "emg")

		eeg_json = {
			"TaskName": bids_path.task,
			"Manufacturer": manufacturer,
			"SamplingFrequency": sfreq,
			"PowerLineFrequency": float(line_freq) if line_freq is not None else "n/a",
			"RecordingDuration": duration,
			"RecordingType": "continuous",
			"EEGReference": "n/a",
			"EEGGround": "n/a",
			"EEGChannelCount": eeg_count,
			"EOGChannelCount": eog_count,
			"ECGChannelCount": ecg_count,
			"EMGChannelCount": emg_count,
		}

		with open(eeg_json_path, "w", encoding="utf-8") as f:
			json.dump(eeg_json, f, indent=2)

	def _write_channels_tsv(self, raw: mne.io.BaseRaw, bids_path: ManualBIDSPath) -> None:
		channels_path = self._sidecar_path(bids_path, "channels.tsv")
		channel_types = raw.get_channel_types()
		units_map = raw._orig_units if hasattr(raw, "_orig_units") else {}
		sfreq = float(raw.info["sfreq"])

		with open(channels_path, "w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f, delimiter="\t")
			writer.writerow(["name", "type", "units", "sampling_frequency", "status", "status_description"])
			for ch_name, ch_type in zip(raw.ch_names, channel_types):
				writer.writerow(
					[
						ch_name,
						ch_type.upper(),
						units_map.get(ch_name, "n/a"),
						f"{sfreq:.6f}",
						"good",
						"",
					]
				)

	def _write_events_files(self, raw: mne.io.BaseRaw, bids_path: ManualBIDSPath) -> None:
		events_tsv_path = self._sidecar_path(bids_path, "events.tsv")
		events_json_path = self._sidecar_path(bids_path, "events.json")
		sfreq = float(raw.info["sfreq"])

		with open(events_tsv_path, "w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f, delimiter="\t")
			writer.writerow(["onset", "duration", "trial_type", "value", "sample"])
			for onset, duration, trial_type in zip(
				raw.annotations.onset,
				raw.annotations.duration,
				raw.annotations.description,
			):
				onset = float(onset)
				duration = float(duration)
				trial_type = str(trial_type)
				writer.writerow([
					f"{onset:.6f}",
					f"{duration:.6f}",
					trial_type,
					trial_type,
					int(round(onset * sfreq)),
				])

		events_json = {
			"onset": {"Description": "Event onset in seconds from the beginning of the recording."},
			"duration": {"Description": "Event duration in seconds."},
			"trial_type": {"Description": "Event category/label."},
			"value": {"Description": "Original event label."},
			"sample": {"Description": "Event onset sample index at recording sampling frequency."},
		}
		with open(events_json_path, "w", encoding="utf-8") as f:
			json.dump(events_json, f, indent=2)

	def write_readme(self, dataset_name: str, source_description: str) -> None:
		if self.dry_run:
			return

		readme_path = self.root / "README"
		text = (
			f"{dataset_name}\n"
			f"Converted to BIDS EEG layout manually by SeizureTransformer tools.\n"
			f"Source: {source_description}\n"
			f"Generated: {datetime.utcnow().isoformat()}Z\n"
		)
		with open(readme_path, "w", encoding="utf-8") as f:
			f.write(text)
