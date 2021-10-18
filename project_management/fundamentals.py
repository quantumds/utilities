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
