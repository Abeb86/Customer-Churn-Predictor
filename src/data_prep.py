import pandas as pd
import numpy as np 


def load_and_clean_data(filePath):
    data = pd.read_csv(filePath)
    # Basic cleaning: drop duplicates and handle missing values
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    return data