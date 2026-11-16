# Enrichment for All

## What this repository is

This repository holds the scripts and notebooks for the Enrichment for All project. They analyse the datasets in `data_final`.

That folder lives in the project drive (INNOCENTI - ENRICHMENT), not in this repository. `data_final` is listed in `.gitignore`, so it is not on GitHub. Before you run any notebook, download the folder and save it in your local copy of this repository.

Keep the file names as they are. The notebooks and `functions.py` open each file by its current name. Renaming a file will make those scripts fail.

## Repository setup

1. Download this repository.
2. Download the `data_final` folder from the drive.
3. Save `data_final` in the root of the local repository, next to this README.
4. Open the repository in your editor. This project has been developed in Cursor.
5. In the terminal, from the repository root, create the virtual environment, activate it, and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.venv` is not in the repository, so you create it once on your own machine. If it already exists, skip the first command and run the other two.

6. Start using the repository.

## Notebooks and scripts

1. `deprivation_analysis.ipynb` compares deprivation scores with indicators from the GrantNav360 platform.
2. `FOI_analysis.ipynb` analyses the data extracted from freedom-of-information (FOI) requests.
3. `pre_survey_analysis.ipynb` runs the descriptive analysis of `all_data_final.csv`. That covers GrantNav360 and Navigate. Arts Council data is not used.
4. `survey_sample_size_usage.ipynb` calculates sample sizes. Collecting survey data was part of the project, but that work stream did not go ahead. The notebook is kept in case a survey becomes a milestone again.
5. `functions.py` holds every function used by the notebooks. Each notebook imports this script. Do not remove it. The rest of the repository depends on it.

## How the repository is organised

Notebooks 1 to 4 should only call functions. Anything that builds a calculation or a chart belongs in `functions.py`. Do not write new functions inside the analysis notebooks.
