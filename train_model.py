"""
Complete Training Pipeline for Rossmann Sales Forecasting
Integrates data preprocessing, feature engineering, and model training
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.append('src')

from data_preprocessing import preprocess_pipeline, basic_data_info
from feature_engineering import feature_engineering_pipeline, get_feature_info
from model_training import ModelTrainer, complete_training_pipeline

def main():
    """
    Complete training pipeline from raw data to trained model
    """
    print("=" * 60)
    print("ROSSMANN SALES FORECASTING - COMPLETE TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Data Preprocessing
    print("\n🔄 STEP 1: DATA PREPROCESSING")
    print("-" * 40)
    
    try:
        # Load and preprocess data
        train_data, test_data, store_data = preprocess_pipeline()
        
        # Show data info
        print("\n📊 Dataset Information:")
        basic_data_info(train_data, test_data, store_data)
        
        print("✅ Data preprocessing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        return False
    
    # Step 2: Feature Engineering
    print("\n🔧 STEP 2: FEATURE ENGINEERING")
    print("-" * 40)
    
    try:
        # Apply feature engineering to training data
        train_features = feature_engineering_pipeline(train_data)
        
        # Apply same transformations to test data (for later use)
        test_features = feature_engineering_pipeline(test_data)
        
        # Show feature info
        print("\n📈 Feature Information:")
        get_feature_info(train_features)
        
        print("✅ Feature engineering completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return False
    
    # Step 3: Model Training
    print("\n🤖 STEP 3: MODEL TRAINING")
    print("-" * 40)
    
    try:
        # Remove any rows with missing target values
        train_clean = train_features.dropna(subset=['Sales'])
        print(f"Training data: {len(train_clean):,} rows after cleaning")
        
        # Complete training pipeline
        trainer = complete_training_pipeline(train_clean, target_column='Sales')
        
        print("✅ Model training completed successfully!")
        
        # Display feature importance
        print("\n🎯 TOP 10 FEATURE IMPORTANCE:")
        feature_importance = trainer.get_feature_importance()
        print(feature_importance.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return False
    
    # Step 4: Final Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📁 Generated Files:")
    print("- models/xgboost_model.pkl (trained XGBoost model)")
    print("- models/scaler.pkl (feature scaler)")
    print("- models/encoder.pkl (categorical encoders)")
    
    print("\n🚀 Next Steps:")
    print("1. Test predictions on validation data")
    print("2. Create Streamlit dashboard")
    print("3. Deploy to Streamlit Cloud")
    print("4. Add to your portfolio!")
    
    return True


def quick_prediction_test():
    """
    Quick test of the complete pipeline with predictions
    """
    print("\n🧪 TESTING PREDICTION PIPELINE")
    print("-" * 40)
    
    try:
        # Load test data (reuse preprocessing)
        _, test_data, _ = preprocess_pipeline()
        test_features = feature_engineering_pipeline(test_data)
        
        # Load trained model
        trainer = ModelTrainer()
        if trainer.load_models():
            
            # Make predictions on first 10 test samples
            sample_data = test_features.head(10)
            predictions = trainer.predict(sample_data)
            
            print(f"✅ Successfully generated {len(predictions)} predictions!")
            print("Sample predictions:", predictions[:5].round(2))
            
            return True
        else:
            print("❌ Could not load trained models")
            return False
            
    except Exception as e:
        print(f"❌ Error in prediction test: {e}")
        return False


if __name__ == "__main__":
    # Run complete training pipeline
    success = main()
    
    if success:
        # Test prediction pipeline
        quick_prediction_test()
        
        print("\n" + "=" * 60)
        print("🎯 PROJECT STATUS: READY FOR STREAMLIT DASHBOARD!")
        print("=" * 60)
    else:
        print("\n❌ Training pipeline failed. Please check error messages above.")