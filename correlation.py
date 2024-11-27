import pandas as pd
from extractor import extract_features

def calculate_correlation(dataset):
    df = pd.DataFrame([extract_features(email) for email in dataset.get("text")])
    return df.corr()

