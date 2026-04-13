"""
Download dan persiapkan dataset Credit Risk untuk proyek MSML.
Dataset: Lending Club Loan Data (simplified credit scoring)
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import os

def download_credit_dataset():
    """Download dataset credit-g dari OpenML (German Credit Dataset)"""
    print("Downloading German Credit Dataset from OpenML...")
    data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
    df = data.frame
    
    # Simpan raw dataset
    output_path = os.path.join(os.path.dirname(__file__), "credit_risk_raw.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target distribution:\n{df['class'].value_counts()}")
    return df

if __name__ == "__main__":
    download_credit_dataset()
