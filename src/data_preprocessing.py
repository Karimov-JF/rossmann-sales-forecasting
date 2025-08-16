"""
Data Preprocessing Module for Rossmann Sales Forecasting
Handles data loading, cleaning, and basic preprocessing
"""

import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the Rossmann dataset files
    
    Returns:
        Tuple of (train_data, test_data, store_data)
    """
    try:
        # Load the datasets
        train_data = pd.read_csv('data/train.csv')
        test_data = pd.read_csv('data/test.csv') 
        store_data = pd.read_csv('data/store.csv')
        
        print(f"✅ Data loaded successfully!")
        print(f"   - Train data: {train_data.shape}")
        print(f"   - Test data: {test_data.shape}")
        print(f"   - Store data: {store_data.shape}")
        
        return train_data, test_data, store_data
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find data files. {e}")
        print("Make sure train.csv, test.csv, and store.csv are in the 'data/' directory")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def merge_with_store_data(sales_data: pd.DataFrame, store_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sales data with store information
    
    Args:
        sales_data: Train or test sales data
        store_data: Store information data
        
    Returns:
        Merged dataframe
    """
    merged_data = sales_data.merge(store_data, on='Store', how='left')
    
    print(f"✅ Merged data shape: {merged_data.shape}")
    return merged_data

def filter_open_stores(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only include data from open stores (Open = 1)
    Note: Only applies to training data as test data doesn't have 'Open' column
    
    Args:
        data: Sales data with store information
        
    Returns:
        Filtered dataframe
    """
    if 'Open' in data.columns:
        # Filter to open stores only
        filtered_data = data[data['Open'] == 1].copy()
        print(f"✅ Filtered to open stores: {filtered_data.shape}")
        return filtered_data
    else:
        print(f"✅ No 'Open' column found (test data): {data.shape}")
        return data

def fill_competition_distance(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing CompetitionDistance values
    
    Args:
        data: DataFrame with potential missing CompetitionDistance
        
    Returns:
        DataFrame with filled values
    """
    data_filled = data.copy()
    
    # Fill missing CompetitionDistance with a large number (50000)
    missing_count = data_filled['CompetitionDistance'].isna().sum()
    if missing_count > 0:
        data_filled['CompetitionDistance'].fillna(50000, inplace=True)
        print(f"✅ Filled {missing_count} missing CompetitionDistance values")
    else:
        print("✅ No missing CompetitionDistance values")
    
    return data_filled

def preprocess_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline
    
    Returns:
        Tuple of (processed_train_data, processed_test_data, store_data)
    """
    print("🔄 Starting data preprocessing pipeline...")
    
    # Step 1: Load data
    train_data, test_data, store_data = load_data()
    
    # Step 2: Merge with store data
    print("\n📊 Merging with store data...")
    train_merged = merge_with_store_data(train_data, store_data)
    test_merged = merge_with_store_data(test_data, store_data)
    
    # Step 3: Filter open stores (only for training data)
    print("\n🏪 Filtering open stores...")
    train_filtered = filter_open_stores(train_merged)
    test_filtered = test_merged  # Test data doesn't need filtering
    
    # Step 4: Fill missing values
    print("\n🔧 Filling missing values...")
    train_final = fill_competition_distance(train_filtered)
    test_final = fill_competition_distance(test_filtered)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   - Final train data: {train_final.shape}")
    print(f"   - Final test data: {test_final.shape}")
    
    # Return all three: processed train, processed test, original store data
    return train_final, test_final, store_data

def basic_data_info(train_data: pd.DataFrame, test_data: pd.DataFrame, store_data: pd.DataFrame):
    """
    Print basic information about the datasets
    
    Args:
        train_data: Training dataset
        test_data: Test dataset  
        store_data: Store dataset
    """
    print("\n📈 DATASET INFORMATION")
    print("=" * 40)
    
    print(f"Training Data: {train_data.shape}")
    print(f"Test Data: {test_data.shape}")
    print(f"Store Data: {store_data.shape}")
    
    print(f"\nTraining Data Columns: {list(train_data.columns)}")
    print(f"Test Data Columns: {list(test_data.columns)}")
    
    # Check for missing values
    print(f"\nMissing Values in Training Data:")
    train_missing = train_data.isnull().sum()
    for col, missing in train_missing.items():
        if missing > 0:
            print(f"   - {col}: {missing}")
    
    print(f"\nMissing Values in Test Data:")
    test_missing = test_data.isnull().sum()
    for col, missing in test_missing.items():
        if missing > 0:
            print(f"   - {col}: {missing}")
    
    # Basic statistics
    if 'Sales' in train_data.columns:
        print(f"\nSales Statistics:")
        print(f"   - Mean: {train_data['Sales'].mean():.2f}")
        print(f"   - Median: {train_data['Sales'].median():.2f}")
        print(f"   - Min: {train_data['Sales'].min():.2f}")
        print(f"   - Max: {train_data['Sales'].max():.2f}")

if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing Data Preprocessing Module")
    print("=" * 50)
    
    try:
        train_data, test_data, store_data = preprocess_pipeline()
        basic_data_info(train_data, test_data, store_data)
        print("\n🎉 Preprocessing module working correctly!")
    except Exception as e:
        print(f"\n❌ Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()