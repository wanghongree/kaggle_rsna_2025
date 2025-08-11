# Kaggle RSNA Intracranial Aneurysm Detection project

## Project structure
kaggle_rsna_2025/
├── data/
├── ignore/
├── scripts/
├── reports/
├── config/
├── requirements.txt

 - kaggle_rsna_2025: the working folder
 - data/: raw/, interim/, processed/
 - scripts/: folder contains all analysis script
 - modules/: reusable functions
 - reports/ — figures, tables, markdown/HTML reports
 - ignore/: folder that's irrelavent to your work
 - config/: YAML/JSON config
 - requirements.txt: python packages

# Dataset Summary
## Overview
The dataset contains thousands of brain imaging series from various modalities (CTA, MRA, MRI T1 post-contrast, T2-weighted), labeled for aneurysm presence across 13 anatomical locations. Spatial localization labels and vessel segmentations are provided for a subset of cases.


## Folder Structure
data/
raw/
  ├── train.csv
  ├── train_localizers.csv
  ├── series/{SeriesInstanceUID}/{SOPInstanceUID}.dcm
  ├── segmentations/{SeriesInstanceUID}/
  │   ├── {SeriesInstanceUID}.nii
  │   └── {SeriesInstanceUID}_cowseg.nii
  └── kaggle_evaluation/   # not relevant for training
interim/
processed/

## train.csv
Primary training labels — one row per image series.

**Columns:**
- `SeriesInstanceUID` — Unique scan series identifier.
- `PatientAge` *(int)* — Age in years.
- `PatientSex` *(str)* — M/F.
- `Modality` — Imaging modality (CTA, MRA, MRI, etc.).
- 13 binary columns for aneurysm presence in specific arteries:
  1. Left Infraclinoid Internal Carotid Artery  
  2. Right Infraclinoid Internal Carotid Artery  
  3. Left Supraclinoid Internal Carotid Artery  
  4. Right Supraclinoid Internal Carotid Artery  
  5. Left Middle Cerebral Artery  
  6. Right Middle Cerebral Artery  
  7. Anterior Communicating Artery  
  8. Left Anterior Cerebral Artery  
  9. Right Anterior Cerebral Artery  
  10. Left Posterior Communicating Artery  
  11. Right Posterior Communicating Artery  
  12. Basilar Tip  
  13. Other Posterior Circulation
- `Aneurysm Present` — Main binary target (1 = aneurysm present anywhere).

## train_localizers.csv
Localization of aneurysms at image level.

**Columns:**
- `SeriesInstanceUID` — Links to train.csv.
- `SOPInstanceUID` — Image-level identifier within the series.
- `coordinates` — XY center coordinates of the aneurysm.
- `location` — Text description of anatomical location.


## series/
DICOM files grouped by `SeriesInstanceUID`.

Format:
series/{SeriesInstanceUID}/{SOPInstanceUID}.dcm

## segmentations/
NIfTI vessel segmentations for some series.

**Label Map:**
| Value | Vessel |
|---|---|
| 1  | Other Posterior Circulation |
| 2  | Basilar Tip |
| 3  | Right Posterior Communicating Artery |
| 4  | Left Posterior Communicating Artery |
| 5  | Right Infraclinoid Internal Carotid Artery |
| 6  | Left Infraclinoid Internal Carotid Artery |
| 7  | Right Supraclinoid Internal Carotid Artery |
| 8  | Left Supraclinoid Internal Carotid Artery |
| 9  | Right Middle Cerebral Artery |
| 10 | Left Middle Cerebral Artery |
| 11 | Right Anterior Cerebral Artery |
| 12 | Left Anterior Cerebral Artery |
| 13 | Anterior Communicating Artery |

---

## kaggle_evaluation/
Contains files used by the competition API for serving the test set (≈2500 series). Not needed for model training.


# “Do Not Modify Lists
avoid modifying the following: 
 - data/raw: raw data, big files
 - venv: virtual enviroment; no need to read
 - ignore: unrelavent for your work; no need to read



# how claude should work here
 - Default editing targets (in order):
    - 1 modules/*.py, 2 scripts/*.py, 3 configs
 - Prefer refactors into modules/ over writing long code in scripts/.
 - Keep analysis cells short (≤ ~30 lines) and focused on one task.
 - Ask before adding new dependencies or changing project layout.
 - Persist outputs only to data/interim, data/processed, reports/figs, reports/tables.
 - Re-runnability: every script must run top-to-bottom without manual tweaks.
 - virtual enviroment called venv
 - update new packages to requirements.txt
 - Never write to data/raw/. Writes go to interim/ or processed/.

#  Commands
 - install: pip install -r requirements.txt
 - use Python “Run Cell” (# %%) for sections.


# Coding Style & Conventions
## General
 - Python ≥ 3.10, type hints required in modules/.
 - Docstrings: Google style.
 - Small, single-purpose functions. Prefer pure functions; isolate I/O.
 - Pandas: favor vectorized ops; avoid loops; chain with .pipe() for clarity.
 - Plotting: centralize common styles in modules/viz.py; save figures to reports/figs/.
 - Randomness: set seeds via numpy.random.default_rng(seed); keep seed in config.

## Naming
 - Modules: snake_case.py; functions: verb_noun (e.g., load_parquet, clean_dates).
 - Scripts: prefix with order NN_ + task (e.g., 02_feature_summary.py).

## ariables:
 - DataFrames: df, df_users; series: s; paths: path_*.

## Imports
 - Standard library → third-party → local, separated by blank lines.
 - No wildcard imports. Prefer explicit: from modules.clean import normalize_names.

## Error handling & logging
 - Raise precise exceptions in modules/; don’t silently pass.
 - Use logging (module-level logger) for info/warn/debug; avoid print in modules/.


## test
 - Unit tests for non-trivial functions in tests/ in a corrsponding script.
 - Fast tests only (seconds). Heavy I/O mocked or sample fixtures in tests/fixtures/.
 - when developing a script, using a sample data or mocked data rather than full data, as full data is huge. 


# debug
 - scripts and outputs for debug should go to debug/ folder
 - run python scripts using virtual enviroment venv







