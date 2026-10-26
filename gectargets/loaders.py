"""
Dataset loaders.

A loader only has to answer two questions: which subjects are there (and in
which group), and what are a subject's epoched time series. Everything else
(filtering, targets, fits) is shared.

To add your own data either
  * store it in the layout read by ``BidsEpochsLoader`` (no code needed), or
  * subclass ``DatasetLoader`` and register it with ``@register("name")``.
See docs/adding_your_data.md.
"""
import zlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

LOADERS = {}


def register(name):
    """Class decorator: make a loader available as ``"loader": name``."""
    def wrap(cls):
        LOADERS[name] = cls
        return cls
    return wrap


def get_loader(dataset_cfg):
    name = dataset_cfg.get("loader", "bids_epochs")
    if name not in LOADERS:
        raise KeyError(f"Unknown loader '{name}'. Registered: {list(LOADERS)}")
    return LOADERS[name](dataset_cfg)


class DatasetLoader:
    """Base class. Subclasses implement ``subjects``, ``load_epochs`` and set
    ``self.sfreq`` (Hz; for fMRI 1 / TR)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sfreq = cfg.get("sfreq")
        self.n_epochs = cfg.get("n_epochs")
        self.epochs_seed = cfg.get("epochs_seed", 0)

    def subjects(self):
        """dict {subject_id: group label}."""
        raise NotImplementedError

    def load_epochs(self, subject):
        """Array (n_epochs, n_nodes, n_times), or None if missing."""
        raise NotImplementedError

    def get_epochs(self, subject):
        """``load_epochs`` with the optional fixed number of epochs.

        With ``n_epochs: N`` in the dataset config every subject contributes N
        epochs drawn at random (seeded by ``epochs_seed`` and the subject id,
        so always the same ones); subjects with fewer epochs are skipped.
        The size of the noise in FS (and so of its antisymmetric part) falls
        with the number of epochs, so a fixed N keeps subjects comparable.
        """
        epochs = self.load_epochs(subject)
        if epochs is None or self.n_epochs is None:
            return epochs
        if len(epochs) < self.n_epochs:
            print(f"{subject}: {len(epochs)} epochs < n_epochs = "
                  f"{self.n_epochs}, skipped")
            return None
        rng = np.random.default_rng(
            [self.epochs_seed, zlib.crc32(subject.encode())])
        idx = np.sort(rng.choice(len(epochs), self.n_epochs, replace=False))
        return epochs[idx]


@register("bids_epochs")
class BidsEpochsLoader(DatasetLoader):
    """Epoched data in the fdt_eeg layout (UCM MEG, UniRoma EEG, ADNI fMRI)::

        <root>/participants.tsv          participant_id, <group_column>, ...
        <root>/derivatives/<epochs_dir>/<prefix><id>/<modality>/ses-<session>/
            <prefix><id>_ses-<session>_eegEpochs.h5   dataset "eegData",
                                                      (n_epochs, n_nodes, n_times)
            <prefix><id>_ses-<session>_mtgInfo.csv    columns Label, fsample

    Config keys: root, epochs_dir, modality, session, subject_prefix (default
    ""), group_column (default "classification"), participants_filter
    ({column: [allowed values]}), sfreq (default: read from mtgInfo).
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.root = Path(cfg["root"])
        self.epochs_path = self.root / "derivatives" / cfg["epochs_dir"]
        self.prefix = cfg.get("subject_prefix", "")
        df = pd.read_csv(self.root / "participants.tsv", sep="\t")
        for column, allowed in cfg.get("participants_filter", {}).items():
            df = df[df[column].isin(allowed)]
        self.participants = df
        if self.sfreq is None:
            self.sfreq = self._read_sfreq()

    def subjects(self):
        col = self.cfg.get("group_column", "classification")
        return dict(zip(self.participants["participant_id"].astype(str),
                        self.participants[col].astype(str)))

    def _file(self, subject, suffix):
        sub = f"{self.prefix}{subject}"
        ses = self.cfg["session"]
        return (self.epochs_path / sub / self.cfg["modality"] / f"ses-{ses}"
                / f"{sub}_ses-{ses}_{suffix}")

    def _read_sfreq(self):
        for subject in self.subjects():
            path = self._file(subject, "mtgInfo.csv")
            if path.exists():
                return float(pd.read_csv(path)["fsample"].iloc[0])
        raise FileNotFoundError(
            f"No mtgInfo.csv found under {self.epochs_path}; set 'sfreq' in "
            f"the dataset config.")

    def load_epochs(self, subject):
        path = self._file(subject, "eegEpochs.h5")
        if not path.exists():
            print(f"{subject}: no data at {path}, skipped")
            return None
        with h5py.File(path, "r") as f:
            return np.asarray(f["eegData"], dtype=float)
