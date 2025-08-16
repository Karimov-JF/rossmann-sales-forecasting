"""
Data preprocessing module for Rossmann Sales Forecasting
Handles data loading, cleaning, and initial transformations
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os
import sys


def load_data(data_path: str = './data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, test, and store data from CSV files
    
    Args:
        data_path: Path to data directory containing CSV files
        
    Returns:
        Tuple of (train_df, test_df, store_df)
        
    Raises:
        FileNotFoundError: If required CSV files are not found
    """
    try:
        train_path = os.path.join(data_path, 'train.csv')
        test_path = os.path.join(data_path, 'test.csv')
        store_path = os.path.join(data_path, 'store.csv')
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        store_df = pd.read_csv(store_path)
        
        print(f"✅ Data loaded successfully:")
        print(f"   Train: {train_df.shape}")
        print(f"   Test: {test_df.shape}")
        print(f"   Store: {store_df.shape}")
        
        return train_df, test_df, store_df
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print(f"   Make sure CSV files exist in {data_path}/")
        raise


def merge_with_store_data(df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sales data with store information
    
    Args:
        df: Sales dataframe (train or test)
        store_df: Store information dataframe
        
    Returns:
        Merged dataframe with store information
    """
    merged_df = df.merge(store_df, on='Store', how='left')
    
    print(f"✅ Merged data shape: {merged_df.shape}")
    print(f"   Added store features: {list(store_df.columns[1:])}")  # Exclude 'Store' column
    
    return merged_df


def filter_open_stores(df: pd.DataFrame, keep_open_only: bool = True) -> pd.DataFrame:
    """
    Filter stores based on Open status
    
    Args:
        df: Dataframe with 'Open' column
        keep_open_only: If True, keep only open stores (Open=1)
        
    Returns:
        Filtered dataframe
    """
    if 'Open' not in df.columns:
        print("⚠️  Warning: 'Open' column not found, returning original dataframe")
        return df
    
    if keep_open_only:
        original_count = len(df)
        df_filtered = df[df['Open'] == 1].copy()
        removed_count = original_count - len(df_filtered)
        
        print(f"✅ Filtered to open stores only:")
        print(f"   Removed {removed_count:,} closed store records")
        print(f"   Remaining: {len(df_filtered):,} records")
        
        return df_filtered
    
    return df


def fill_competition_distance(df: pd.DataFrame, fill_value: float = 50000) -> pd.DataFrame:
    """
    Fill missing CompetitionDistance values
    
    Args:
        df: Dataframe with CompetitionDistance column
        fill_value: Value to fill missing distances (default: 50000)
        
    Returns:
        Dataframe with filled CompetitionDistance
    """
    if 'CompetitionDistance' not in df.columns:
        print("⚠️  Warning: 'CompetitionDistance' column not found")
        return df
    
    missing_count = df['CompetitionDistance'].isna().sum()
    
    if missing_count > 0:
        df = df.copy()
        df['CompetitionDistance'].fillna(fill_value, inplace=True)
        print(f"✅ Filled {missing_count:,} missing CompetitionDistance values with {fill_value}")
    else:
        print("✅ No missing CompetitionDistance values found")
    
    return df


def basic_data_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print basic information about the dataset
    
    Args:
        df: Dataframe to analyze
        name: Name for display purposes
    """
    print(f"\n📊 {name} Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"   Missing values:")
        for col, count in missing[missing > 0].items():
            print(f"     {col}: {count:,} ({count/len(df)*100:.1f}%)")
    else:
        print(f"   ✅ No missing values")
    
    # Data types
    print(f"   Data types: {dict(df.dtypes.value_counts())}")


def preprocess_pipeline(data_path: str = './data', 
                       keep_open_only: bool = True,
                       competition_fill_value: float = 50000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline for Rossmann data
    
    Args:
        data_path: Path to data directory
        keep_open_only: Whether to filter only open stores
        competition_fill_value: Value to fill missing competition distances
        
    Returns:
        Tuple of (processed_train_df, processed_test_df)
    """
    print("🚀 Starting Rossmann data preprocessing pipeline...")
    
    # Step 1: Load data
    train_df, test_df, store_df = load_data(data_path)
    
    # Step 2: Basic info
    basic_data_info(train_df, "Training Data")
    basic_data_info(test_df, "Test Data")
    basic_data_info(store_df, "Store Data")
    
    # Step 3: Merge with store data
    train_merged = merge_with_store_data(train_df, store_df)
    test_merged = merge_with_store_data(test_df, store_df)
    
    # Step 4: Filter open stores (only for training data)
    if keep_open_only and 'Open' in train_merged.columns:
        train_processed = filter_open_stores(train_merged, keep_open_only=True)
    else:
        train_processed = train_merged.copy()
    
    # For test data, we typically keep all records
    test_processed = test_merged.copy()
    
    # Step 5: Fill missing competition distances
    train_processed = fill_competition_distance(train_processed, competition_fill_value)
    test_processed = fill_competition_distance(test_processed, competition_fill_value)
    
    print(f"\n✅ Preprocessing pipeline completed!")
    print(f"   Final train shape: {train_processed.shape}")
    print(f"   Final test shape: {test_processed.shape}")
    
    return train_processed, test_processed


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Rossmann data preprocessing...")
    
    # First, let's check current directory and data path
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if data directory exists
    data_path = './data'
    if os.path.exists(data_path):
        print(f"✅ Data directory found: {os.path.abspath(data_path)}")
        files = os.listdir(data_path)
        print(f"   Files in data directory: {files}")
    else:
        print(f"❌ Data directory not found: {os.path.abspath(data_path)}")
        print("   Trying current directory...")
        data_path = './data'
        if os.path.exists(data_path):
            print(f"✅ Found data directory: {os.path.abspath(data_path)}")
            files = os.listdir(data_path)
            print(f"   Files: {files}")
        else:
            print(f"❌ Data directory not found in current directory either")
            print("   Please check your data directory location")
            sys.exit(1)  # Exit the script instead of return
    
    try:
        # Run the complete pipeline
        print("\n🚀 Running preprocessing pipeline...")
        train_df, test_df = preprocess_pipeline(data_path)
        
        print(f"\n🎯 Pipeline Results:")
        print(f"   Train data: {train_df.shape}")
        print(f"   Test data: {test_df.shape}")
        print(f"   Available columns: {list(train_df.columns)}")
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        print("   Check your data files and paths")