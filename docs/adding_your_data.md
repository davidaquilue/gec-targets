# Adding your own time series

The pipeline needs, for each subject, an array of **epoched time series**
`(n_epochs, n_nodes, n_times)` and a **group label**. Nodes can be EEG
channels, MEG sources or fMRI parcels. There are two ways in.

## Option A: store the data in the expected layout (no code)

The built-in `bids_epochs` loader reads

```
my_dataset/
├── participants.tsv                      # tab-separated
└── derivatives/
    └── <epochs_dir>/
        └── <subject>/<modality>/ses-<session>/
            ├── <subject>_ses-<session>_eegEpochs.h5   # dataset "eegData"
            └── <subject>_ses-<session>_mtgInfo.csv    # columns Label, fsample
```

`participants.tsv` needs a `participant_id` column and a group column
(default `classification`):

```
participant_id	classification	age
sub-01	HC	71
sub-02	AD	68
```

### Step 1: write the files

```python
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

root = Path("my_dataset")
sfreq = 250.0                                   # Hz (fMRI: 1 / TR)
rows = []
for sub, group, x in my_subjects:               # x: (n_epochs, n_nodes, n_times)
    d = root / "derivatives" / "epochs_mine" / sub / "eeg" / "ses-rest"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / f"{sub}_ses-rest_eegEpochs.h5", "w") as f:
        f.create_dataset("eegData", data=x)
    pd.DataFrame({"Label": [f"node_{i}" for i in range(x.shape[1])],
                  "fsample": sfreq}).to_csv(d / f"{sub}_ses-rest_mtgInfo.csv",
                                            index=False)
    rows.append({"participant_id": sub, "classification": group})
pd.DataFrame(rows).to_csv(root / "participants.tsv", sep="\t", index=False)
```

`examples/make_synthetic_dataset.py` does exactly this with simulated data.

A continuous recording is one epoch: `x[None]`, shape `(1, n_nodes, n_times)`.
Split-half analyses then compare the first and the second half of the
recording.

### Step 2: declare the dataset in your config

```json
"datasets": {
  "MyData": {
    "loader": "bids_epochs",
    "root": "/path/to/my_dataset",
    "epochs_dir": "epochs_mine",
    "modality": "eeg",
    "session": "rest",
    "band": "alpha",
    "sc_path": null
  }
}
```

* `band`: a name from the `bands` block (filtered with a zero-phase
  Butterworth), or `null` to use the data as stored (e.g. already band-passed
  fMRI).
* `sc_path`: an `(n_nodes, n_nodes)` structural connectivity matrix (`.csv`,
  `.txt` or `.npy`) in the same node order as the data. It sets the initial
  couplings and which ones can be non-zero. `null` (e.g. sensor-space EEG)
  starts from a seeded random matrix; set `n_init` in `fitting` (or in the
  sweep `grid`) to average several random starts.
* Optional: `subject_prefix` (if folders are `sub-<id>` but
  `participant_id` is `<id>`), `group_column`, `participants_filter`
  (e.g. `{"classification": ["HC", "AD"]}`), `sfreq` (otherwise read from
  `mtgInfo.csv`), `n_epochs` and `epochs_seed` (use the same random N epochs
  for every subject; subjects with fewer are skipped).

### Step 3: add it to the sweep and run

```json
"sweep": [{"datasets": ["MyData"], "models": ["network"]}]
```

```bash
python -m gectargets tasks   my_config.json
python -m gectargets targets my_config.json
python -m gectargets fit     my_config.json --all
```

## Option B: write a loader

If your data live in another format, subclass `DatasetLoader` and register
it. Put the class in a module that is imported before the pipeline runs (for
example add it to `gectargets/loaders.py`).

```python
import numpy as np
from gectargets.loaders import DatasetLoader, register


@register("my_npy")
class NpyLoader(DatasetLoader):
    """One file <root>/<subject>.npy per subject, (n_epochs, n_nodes, n_times),
    and <root>/groups.csv with columns subject, group."""

    def __init__(self, cfg):
        super().__init__(cfg)        # reads sfreq, n_epochs, epochs_seed
        self.root = cfg["root"]

    def subjects(self):
        import pandas as pd
        df = pd.read_csv(self.root / "groups.csv")
        return dict(zip(df.subject.astype(str), df.group.astype(str)))

    def load_epochs(self, subject):
        path = self.root / f"{subject}.npy"
        return np.load(path) if path.exists() else None
```

and in the config `"loader": "my_npy"`, `"root": ...`, `"sfreq": 250`. Any
other key you add to the dataset block reaches your loader in `cfg`.

## Checklist

* The node order of the data and of the SC must match.
* Units do not matter (every epoch is z-scored), but each node must vary in
  every epoch (flat channels give NaN correlations).
* `tau` is fitted per dataset and variant (first lag where the mean
  autocorrelation drops below 1/e) and printed by the `targets` step: check
  it is a sensible number of samples for your sampling rate.
