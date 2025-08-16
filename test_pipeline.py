"""
Test script for Rossmann Sales Forecasting Pipeline
Quick verification that all modules work together
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    try:
        from data_preprocessing import load_data
        train_data, test_data, store_data = load_data()
        
        print(f"✅ Train data: {train_data.shape}")
        print(f"✅ Test data: {test_data.shape}")  
        print(f"✅ Store data: {store_data.shape}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing pipeline"""
    print("\nTesting preprocessing pipeline...")
    try:
        from data_preprocessing import preprocess_pipeline
        train_data, test_data, store_data = preprocess_pipeline()
        
        print(f"✅ Preprocessed train: {train_data.shape}")
        print(f"✅ Preprocessed test: {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return None, None

def test_feature_engineering(train_data, test_data):
    """Test feature engineering pipeline"""
    print("\nTesting feature engineering...")
    try:
        from feature_engineering import feature_engineering_pipeline, select_model_features
        
        # Test feature engineering
        train_features = feature_engineering_pipeline(train_data)
        test_features = feature_engineering_pipeline(test_data) 
        
        print(f"✅ Train features: {train_features.shape}")
        print(f"✅ Test features: {test_features.shape}")
        
        # Test feature selection
        selected_features = select_model_features(train_features)
        print(f"✅ Selected features: {selected_features.shape}")
        print(f"✅ Feature columns: {list(selected_features.columns)}")
        
        return train_features
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return None

def test_model_training(train_features):
    """Test model training (quick version)"""
    print("\nTesting model training...")
    try:
        from model_training import ModelTrainer
        
        # Use small sample for quick testing
        sample_size = min(1000, len(train_features))
        train_sample = train_features.sample(sample_size, random_state=42)
        
        # Remove target column and prepare data
        X = train_sample.drop(columns=['Sales'])
        y = train_sample['Sales']
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Prepare features
        X_processed = trainer.prepare_features_for_training(X, is_training=True)
        print(f"✅ Features prepared: {X_processed.shape}")
        
        # Quick training (no hyperparameter tuning)
        results = trainer.train_final_model(X_processed, y)
        print(f"✅ Model trained! RMSE: {results['training_rmse']:.2f}")
        
        # Test prediction
        predictions = trainer.predict(X.head(5))
        print(f"✅ Predictions generated: {predictions[:3].round(2)}")
        
        return True
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def main():
    """Run all tests"""
    print("ROSSMANN PIPELINE TESTING")
    print("=" * 40)
    
    # Test 1: Data Loading
    if not test_data_loading():
        return
    
    # Test 2: Preprocessing  
    train_data, test_data = test_preprocessing()
    if train_data is None:
        return
    
    # Test 3: Feature Engineering
    train_features = test_feature_engineering(train_data, test_data)
    if train_features is None:
        return
    
    # Test 4: Model Training
    if not test_model_training(train_features):
        return
        
    print("\n" + "=" * 40)
    print("🎉 ALL TESTS PASSED!")
    print("Pipeline is working correctly!")
    print("Ready to run full training with train_model.py")
    print("=" * 40)

if __name__ == "__main__":
    main()