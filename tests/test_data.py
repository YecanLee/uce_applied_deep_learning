import pandas as pd
import numpy as np  
import pytest

import os

# Define a function to check the csv files inside the tests folder
# This function will check: 1) if the file has been named as "..._prompts.csv"
# 2) if the file has four columns: "case_number", "evaluation_seed", "prompt"
# 3) if the file's case_number starts from 0 and it is continuous

def test_prompts_csv_files():
    # Define the path to the folder containing the csv files
    path = "tests/"
    # List all the files in the folder
    csv_files = os.listdir(path)
    for csv_file in csv_files:
        # Read the csv file
        df = pd.read_csv(path + csv_file)
        # Check if the file has been named as "..._prompts.csv"
        assert "_prompts.csv" in csv_file, "The prompt file must be named as '..._prompts.csv'"
        # Check if the file has four columns: "case_number", "evaluation_seed", "prompt"
        assert len(df.columns) == 3, "The Stable Diffusion Model requires three columns: 'case_number', 'evaluation_seed', 'prompt'"
        assert "case_number" in df.columns, "case_number column is missing"
        assert "evaluation_seed" in df.columns, "evaluation_seed column is missing"
        assert "prompt" in df.columns, "prompt column is missing"
        # Check if the file's case_number starts from 0 and it is continuous
        assert np.all(df["case_number"] == np.arange(len(df))), "THe case_number column must start from 0 and be continuous"

