"""
Feature engineering module for Rossmann Sales Forecasting
Creates custom features for better model performance
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


def split_date_features(df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
    """
    Extract date components (Year, Month, Day, WeekOfYear) from date column
    
    Args:
        df: Dataframe with date column
        date_column: Name of the date column
        
    Returns:
        Dataframe with additional date features
    """
    if date_column not in df.columns:
        print(f"⚠️  Warning: '{date_column}' column not found")
        return df
    
    df = df.copy()
    
    # Convert to datetime if not already
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract date components
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['Day'] = df[date_column].dt.day
    df['WeekOfYear'] = df[date_column].dt.isocalendar().week
    
    print(f"✅ Created date features: Year, Month, Day, WeekOfYear")
    print(f"   Date range: {df[date_column].min().date()} to {df[date_column].max().date()}")
    
    return df


def create_competition_months_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create CompOpenSince feature (months since competition opened)
    
    Args:
        df: Dataframe with Date, CompetitionOpenSinceMonth, CompetitionOpenSinceYear
        
    Returns:
        Dataframe with CompOpenSince feature
    """
    required_cols = ['Date', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Warning: Missing columns for competition feature: {missing_cols}")
        return df
    
    df = df.copy()
    
    # Convert Date to datetime if not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    def calculate_comp_months(row):
        """Calculate months between current date and competition open date"""
        if pd.isna(row['CompetitionOpenSinceMonth']) or pd.isna(row['CompetitionOpenSinceYear']):
            return 0
        
        try:
            # Create competition open date (using day 1)
            comp_date = datetime(
                int(row['CompetitionOpenSinceYear']), 
                int(row['CompetitionOpenSinceMonth']), 
                1
            )
            
            # Calculate months difference
            current_date = row['Date']
            months_diff = (current_date.year - comp_date.year) * 12 + (current_date.month - comp_date.month)
            
            # Return 0 if competition opened after current date
            return max(0, months_diff)
            
        except (ValueError, OverflowError):
            return 0
    
    # Apply the function
    df['CompOpenSince'] = df.apply(calculate_comp_months, axis=1)
    
    # Basic stats
    non_zero_count = (df['CompOpenSince'] > 0).sum()
    print(f"✅ Created CompOpenSince feature:")
    print(f"   Records with competition data: {non_zero_count:,} ({non_zero_count/len(df)*100:.1f}%)")
    print(f"   Range: {df['CompOpenSince'].min()} to {df['CompOpenSince'].max()} months")
    
    return df


def create_promo2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Promo2 related features: Promo2OpenSince and IsPromo2Month
    
    Args:
        df: Dataframe with Promo2, Promo2SinceWeek, Promo2SinceYear, PromoInterval
        
    Returns:
        Dataframe with Promo2 features
    """
    required_cols = ['Date', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Warning: Missing columns for Promo2 features: {missing_cols}")
        return df
    
    df = df.copy()
    
    # Convert Date to datetime if not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Initialize new features
    df['Promo2OpenSince'] = 0
    df['IsPromo2Month'] = 0
    
    def calculate_promo2_features(row):
        """Calculate Promo2 related features for a single row"""
        if row['Promo2'] == 0 or pd.isna(row['Promo2SinceWeek']) or pd.isna(row['Promo2SinceYear']):
            return 0, 0  # Promo2OpenSince, IsPromo2Month
        
        try:
            # Calculate Promo2OpenSince (weeks since promo2 started)
            # Approximate: week 1 = Jan 1st, each week = 7 days
            promo2_start_date = datetime(
                int(row['Promo2SinceYear']), 
                1, 
                1
            ) + pd.Timedelta(weeks=int(row['Promo2SinceWeek'])-1)
            
            current_date = row['Date']
            weeks_diff = (current_date - promo2_start_date).days // 7
            promo2_open_since = max(0, weeks_diff)
            
            # Calculate IsPromo2Month
            is_promo2_month = 0
            if not pd.isna(row['PromoInterval']):
                # PromoInterval contains months like "Jan,Apr,Jul,Oct"
                promo_months = str(row['PromoInterval']).split(',')
                current_month_name = current_date.strftime('%b')  # Jan, Feb, etc.
                
                if current_month_name in promo_months:
                    is_promo2_month = 1
            
            return promo2_open_since, is_promo2_month
            
        except (ValueError, OverflowError):
            return 0, 0
    
    # Apply the function
    promo2_features = df.apply(calculate_promo2_features, axis=1, result_type='expand')
    df['Promo2OpenSince'] = promo2_features[0]
    df['IsPromo2Month'] = promo2_features[1]
    
    # Basic stats
    promo2_active = (df['Promo2'] == 1).sum()
    promo2_month_active = (df['IsPromo2Month'] == 1).sum()
    
    print(f"✅ Created Promo2 features:")
    print(f"   Stores with Promo2: {promo2_active:,} ({promo2_active/len(df)*100:.1f}%)")
    print(f"   Records in Promo2 months: {promo2_month_active:,} ({promo2_month_active/len(df)*100:.1f}%)")
    print(f"   Promo2OpenSince range: {df['Promo2OpenSince'].min()} to {df['Promo2OpenSince'].max()} weeks")
    
    return df


def select_model_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """
    Select relevant features for modeling
    
    Args:
        df: Dataframe with all features
        include_target: Whether to include Sales column (for training data)
        
    Returns:
        Dataframe with selected features
    """
    # Core features for modeling
    model_features = [
        # Store and date info
        'Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'WeekOfYear',
        
        # Basic promotion and store status
        'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
        
        # Store characteristics
        'StoreType', 'Assortment', 'CompetitionDistance',
        
        # Custom engineered features
        'CompOpenSince', 'Promo2OpenSince', 'IsPromo2Month'
    ]
    
    # Add target if requested
    if include_target and 'Sales' in df.columns:
        model_features.append('Sales')
    
    # Add customers if available (for training data)
    # if 'Customers' in df.columns:
    #     model_features.append('Customers')
    
    # Filter to available columns
    available_features = [col for col in model_features if col in df.columns]
    missing_features = [col for col in model_features if col not in df.columns]
    
    if missing_features:
        print(f"⚠️  Missing features (will be skipped): {missing_features}")
    
    df_selected = df[available_features].copy()
    
    print(f"✅ Selected {len(available_features)} features for modeling:")
    print(f"   Features: {available_features}")
    
    return df_selected


def feature_engineering_pipeline(df: pd.DataFrame, is_training_data: bool = True) -> pd.DataFrame:
    """
    Complete feature engineering pipeline
    
    Args:
        df: Input dataframe (output from preprocessing)
        is_training_data: Whether this is training data (includes Sales column)
        
    Returns:
        Dataframe with engineered features ready for modeling
    """
    print(f"🔧 Starting feature engineering pipeline...")
    print(f"   Input shape: {df.shape}")
    print(f"   Data type: {'Training' if is_training_data else 'Test'}")
    
    # Step 1: Create date features
    df_features = split_date_features(df, 'Date')
    
    # Step 2: Create competition features
    df_features = create_competition_months_feature(df_features)
    
    # Step 3: Create Promo2 features
    df_features = create_promo2_features(df_features)
    
    # Step 4: Select model features
    df_final = select_model_features(df_features, include_target=is_training_data)
    
    print(f"\n✅ Feature engineering completed!")
    print(f"   Final shape: {df_final.shape}")
    print(f"   Features created: CompOpenSince, Promo2OpenSince, IsPromo2Month, Year, Month, Day, WeekOfYear")
    
    return df_final


def get_feature_info(df: pd.DataFrame) -> None:
    """
    Display detailed information about engineered features
    
    Args:
        df: Dataframe with engineered features
    """
    print(f"\n📊 Feature Engineering Summary:")
    print(f"   Total features: {len(df.columns)}")
    print(f"   Total records: {len(df):,}")
    
    # Check for missing values in key features
    key_features = ['CompOpenSince', 'Promo2OpenSince', 'IsPromo2Month']
    for feature in key_features:
        if feature in df.columns:
            unique_vals = df[feature].nunique()
            min_val = df[feature].min()
            max_val = df[feature].max()
            print(f"   {feature}: {unique_vals} unique values, range [{min_val}, {max_val}]")
    
    # Data types
    print(f"\n   Data types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"     {dtype}: {count} columns")


if __name__ == "__main__":
    # Example usage - test with preprocessed data
    print("🧪 Testing feature engineering pipeline...")
    
    # Import preprocessing module
    try:
        from data_preprocessing import preprocess_pipeline
        
        # Get preprocessed data
        print("Loading and preprocessing data...")
        train_df, test_df, store_data = preprocess_pipeline()
        
        # Run feature engineering on training data
        print("\n" + "="*60)
        print("TRAINING DATA FEATURE ENGINEERING")
        print("="*60)
        train_features = feature_engineering_pipeline(train_df, is_training_data=True)
        get_feature_info(train_features)
        
        # Run feature engineering on test data  
        print("\n" + "="*60)
        print("TEST DATA FEATURE ENGINEERING")
        print("="*60)
        test_features = feature_engineering_pipeline(test_df, is_training_data=False)
        get_feature_info(test_features)
        
        print(f"\n🎯 Feature Engineering Results:")
        print(f"   Train features: {train_features.shape}")
        print(f"   Test features: {test_features.shape}")
        print(f"   Ready for model training!")
        
    except ImportError:
        print("❌ Could not import data_preprocessing module")
        print("   Make sure data_preprocessing.py is in the same directory")
    except Exception as e:
        print(f"❌ Error running feature engineering: {e}")