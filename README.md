# ICS 471 Skin Lesion Classification

This repository supports the proposal **Leakage-Aware Classification of Dermoscopic Skin Lesions**. The task is seven-class image classification using HAM10000.

## Project definition

- Input: one RGB dermoscopic image.
- Output: probabilities for seven lesion categories.
- Task: single-label multiclass image classification.
- Validation metric: macro-F1.
- Split: approximately 70% training, 15% validation, and 15% test, grouped by `lesion_id`.

This is an academic experiment. It is not a clinical diagnostic tool.

## Dataset

Download HAM10000 from the official Harvard Dataverse record:

https://doi.org/10.7910/DVN/DBW86T

Place the files under `data/raw/`. The scripts search recursively, so the two official image folders can remain separate.

```text
data/raw/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Generate the milestone figures

```bash
python src/inspect_data.py
```

Outputs:

- `outputs/representative_samples.png`: two real images from each class.
- `outputs/class_distribution.png`: class counts and percentages.
- `outputs/dataset_summary.csv`: summary table.

The script uses seed `471` for the sample grid.

## Create fixed splits

```bash
python src/create_splits.py
```

Outputs:

- `splits/train.csv`
- `splits/validation.csv`
- `splits/test.csv`
- `splits/all_splits.csv`

The script uses `StratifiedGroupKFold` and groups by `lesion_id`. It stops if any lesion appears in more than one partition or any class is missing from a partition.

## Reproducibility rules

- Do not change the split after model training starts.
- Fit preprocessing statistics on training data only.
- Apply augmentation and oversampling to training data only.
- Select models using validation macro-F1.
- Evaluate the selected model on the test set once.

## References

- Tschandl, P., Rosendahl, C., and Kittler, H. *The HAM10000 dataset*. Scientific Data 5, 180161, 2018. https://doi.org/10.1038/sdata.2018.161
- HAM10000 dataset record. Harvard Dataverse. https://doi.org/10.7910/DVN/DBW86T
