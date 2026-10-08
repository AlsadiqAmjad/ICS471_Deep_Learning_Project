# ICS 471 Deep Learning Project

## Project definition

This project investigates leakage-aware classification of dermoscopic skin-lesion images from the HAM10000 dataset.

- **Input:** one RGB dermoscopic image.
- **Output:** probabilities for seven skin-lesion classes.
- **Task:** single-label multiclass image classification.
- **Primary validation metric:** macro-F1.
- **Data split:** training, validation, and test partitions grouped by `lesion_id` to prevent images of the same lesion from crossing partitions.

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

Raw dataset files are not stored in this repository. Download and use the dataset according to the access and licensing terms on its official page.

## Setup

Python 3.12 is recommended.

Create the virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows CMD:

```bat
.venv\Scripts\activate
```

Activate it on WSL or Linux:

```bash
source .venv/bin/activate
```

After activation, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Milestone instructions

The requirements for each milestone are stored in `docs/`:

- [Milestone 1 instructions](docs/Milestone_01_Instructions.txt)
- [Milestone 2 instructions](docs/Milestone_02_Instructions.txt)

Each milestone's code, outputs, and report material should be interpreted according to its corresponding instruction file.

## Reproducibility rules

- Use the fixed random seed defined in the project code.
- Do not change the data split after model training begins.
- Group all splits by `lesion_id` to prevent leakage.
- Fit preprocessing statistics on training data only.
- Apply augmentation and oversampling to training data only.
- Select models using validation macro-F1.
- Keep the test set untouched until final evaluation.
- Record the configuration and results of every experiment.

## References

- Tschandl, P., Rosendahl, C., and Kittler, H. *The HAM10000 dataset*. Scientific Data 5, 180161, 2018. https://doi.org/10.1038/sdata.2018.161
- HAM10000 dataset record. Harvard Dataverse. https://doi.org/10.7910/DVN/DBW86T
