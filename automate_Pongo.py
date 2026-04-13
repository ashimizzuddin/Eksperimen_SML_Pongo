"""
automate_Pongo.py
Script otomatisasi preprocessing data Credit Risk.
Mengkonversi proses eksperimen dari notebook menjadi pipeline otomatis.

Kriteria 1 - Skilled Level
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """
    Tahap 1: Memuat dataset dari file CSV.
    
    Args:
        filepath: Path ke file CSV dataset mentah.
    Returns:
        DataFrame dengan data mentah.
    """
    print(f"[1/6] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"      Shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tahap 2: Menangani missing values.
    
    Strategy:
    - Numeric columns: fill with median
    - Categorical columns: fill with mode
    """
    print("[2/6] Handling missing values...")
    
    missing_before = df.isnull().sum().sum()
    print(f"      Missing values before: {missing_before}")
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    missing_after = df.isnull().sum().sum()
    print(f"      Missing values after: {missing_after}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tahap 3: Menghapus data duplikat.
    """
    print("[3/6] Removing duplicates...")
    
    duplicates = df.duplicated().sum()
    print(f"      Duplicates found: {duplicates}")
    
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"      Shape after removing duplicates: {df.shape}")
    return df


def encode_categorical(df: pd.DataFrame) -> tuple:
    """
    Tahap 4: Encoding data kategorikal menggunakan LabelEncoder.
    
    Returns:
        Tuple of (encoded DataFrame, dict of label encoders)
    """
    print("[4/6] Encoding categorical features...")
    
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    print(f"      Categorical columns: {list(categorical_cols)}")
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"      Encoded '{col}': {len(le.classes_)} classes")
    
    return df, label_encoders


def handle_outliers(df: pd.DataFrame, target_col: str = 'class') -> pd.DataFrame:
    """
    Tahap 5: Deteksi dan penanganan outlier menggunakan metode IQR.
    Hanya pada fitur numerik, tidak termasuk target.
    """
    print("[5/6] Handling outliers using IQR method...")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    total_removed = 0
    initial_shape = df.shape[0]
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            # Cap outliers instead of removing to preserve data
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            total_removed += outliers
    
    print(f"      Outliers capped: {total_removed} values across {len(numeric_cols)} columns")
    return df


def scale_features(df: pd.DataFrame, target_col: str = 'class') -> tuple:
    """
    Tahap 6: Normalisasi fitur menggunakan StandardScaler.
    
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    print("[6/6] Scaling features with StandardScaler...")
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"      Scaled {len(feature_cols)} features")
    return df, scaler


def split_data(df: pd.DataFrame, target_col: str = 'class', 
               test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split data menjadi train dan test set.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")
    print(f"  Target distribution (train): {dict(y_train.value_counts())}")
    print(f"  Target distribution (test):  {dict(y_test.value_counts())}")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(input_path: str, output_dir: str = None) -> tuple:
    """
    Pipeline preprocessing otomatis end-to-end.
    
    Tahapan:
    1. Load data
    2. Handle missing values
    3. Remove duplicates
    4. Encode categorical features
    5. Handle outliers (IQR capping)
    6. Scale features (StandardScaler)
    7. Split data (train/test)
    
    Args:
        input_path: Path ke file CSV dataset mentah.
        output_dir: Directory untuk menyimpan dataset terproses.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("=" * 60)
    print("AUTOMATED PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # 1. Load
    df = load_data(input_path)
    
    # 2. Missing values
    df = handle_missing_values(df)
    
    # 3. Duplicates
    df = remove_duplicates(df)
    
    # 4. Encode
    df, label_encoders = encode_categorical(df)
    
    # 5. Outliers
    df = handle_outliers(df)
    
    # 6. Scale
    df, scaler = scale_features(df)
    
    # 7. Split
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Save preprocessed data
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_path), "preprocessing")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_path = os.path.join(output_dir, "credit_risk_train.csv")
    test_path = os.path.join(output_dir, "credit_risk_test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Save full preprocessed dataset
    full_path = os.path.join(output_dir, "credit_risk_preprocessed.csv")
    df.to_csv(full_path, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"PREPROCESSING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Train data saved: {train_path}")
    print(f"  Test data saved:  {test_path}")
    print(f"  Full data saved:  {full_path}")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Path ke dataset mentah
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, "credit_risk_raw.csv")
    output_dir = os.path.join(script_dir, "preprocessing")
    
    if not os.path.exists(raw_data_path):
        print(f"ERROR: Dataset not found at {raw_data_path}")
        print("Please run prepare_dataset.py first!")
        exit(1)
    
    X_train, X_test, y_train, y_test = preprocess_pipeline(raw_data_path, output_dir)
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")
