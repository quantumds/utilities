Properties should be divided in:

properties_general
properties_datafiles
properties_type_casting

# How to import Properties:
############################################################################
# LIBRARIES
############################################################################
# First the native libraries of Python:
import os
from pathlib import Path
import glob

# Then external libraries:
import yaml
import git
import pandas as pd

############################################################################
# PROPERTIES IMPORT
############################################################################
# We use git library to get the source directory of repo (A part):
repo = git.Repo('.', search_parent_directories=True)
# We create the extension where the "properties.yaml" file exists (B part)
yaml_location = os.path.join("src", "properties.yaml")
# We transform "repo" object which is in git format to Windows path string format:
root = Path(repo.working_tree_dir)
# Connecting A and B parts:
yaml_file = os.path.join(root, yaml_location)

# Import properties.yaml file:
with open(yaml_file) as file:
    properties = yaml.load(file, Loader=yaml.BaseLoader)

############################################################################
# STRUCTURE OF DATA SCIENCE PROJECTS
############################################################################
📁 kaggle-competition-name/
│── 📁 data/                   # All data-related files (input/output)
│   ├── 📂 raw/                # Original data (DO NOT MODIFY)
│   ├── 📂 processed/          # Cleaned & transformed data
│   ├── 📂 external/           # Additional data sources
│   ├── 📂 interim/            # Intermediate data files
│   ├── train.csv              # Training dataset (if applicable)
│   ├── test.csv               # Test dataset (if applicable)
│── 📁 notebooks/              # Jupyter Notebooks for EDA & analysis
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature engineering
│   ├── 03_modeling.ipynb      # Baseline models
│   ├── 04_hyperparameter_tuning.ipynb # Hyperparameter tuning
│   ├── 05_ensemble.ipynb      # Ensembling & stacking
│── 📁 src/                    # Source code for reusable scripts
│   ├── 📂 data/               # Scripts for data processing
│   │   ├── make_dataset.py    # Load & preprocess data
│   │   ├── feature_engineering.py # Feature engineering
│   ├── 📂 models/             # Scripts for model training
│   │   ├── train_model.py     # Model training pipeline
│   │   ├── predict_model.py   # Model inference
│   ├── 📂 utils/              # Utility functions (helpers, metrics, etc.)
│   │   ├── metrics.py         # Custom evaluation metrics
│   │   ├── config.py          # Configuration parameters
│── 📁 experiments/            # Experiment tracking
│   ├── experiment_001.json    # Log of hyperparameters & results
│   ├── experiment_002.json    # Another experiment log
│── 📁 models/                 # Saved model weights & artifacts
│   ├── baseline_model.pkl     # Baseline model
│   ├── final_model.pkl        # Best-performing model
│── 📁 submissions/            # Submission CSVs for Kaggle
│   ├── submission_001.csv     # First submission
│   ├── submission_002.csv     # Improved submission
│── 📁 logs/                   # Training logs
│   ├── training_log.txt       # Log file with training details
│── 📁 reports/                # Documentation & reports
│   ├── report.pdf             # Final report
│   ├── README.md              # Project overview
│── 📁 configs/                # Configuration files
│   ├── config.yaml            # Model and pipeline settings
│── .gitignore                 # Ignore unnecessary files (e.g., data, models)
│── requirements.txt           # Dependencies for the project
│── setup.py                   # Installation script (if applicable)
