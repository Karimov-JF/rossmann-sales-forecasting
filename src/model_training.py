"""
Model Training Module for Rossmann Sales Forecasting
Handles model training, hyperparameter tuning, and model persistence
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Handles all model training operations for Rossmann sales forecasting
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.best_params = None
        self.cv_scores = None
        
        # Features that need encoding
        self.categorical_features = ['StoreType', 'Assortment', 'StateHoliday']
        
        # Features that need scaling (numerical features)
        self.numerical_features = [
            'Store', 'DayOfWeek', 'Customers', 'Promo', 'SchoolHoliday',
            'CompetitionDistance', 'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear', 'Promo2', 'Year', 'Month', 
            'Day', 'WeekOfYear', 'CompOpenSince', 'Promo2OpenSince', 'IsPromo2Month'
        ]
    
    def prepare_features_for_training(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepare features for model training by encoding and scaling
        
        Args:
            df: Input dataframe with features
            is_training: Whether this is training data (fit transformers) or test data (transform only)
            
        Returns:
            Processed dataframe ready for model training
        """
        df_processed = df.copy()
        
        # Handle categorical features with label encoding
        for feature in self.categorical_features:
            if feature in df_processed.columns:
                if is_training:
                    # Fit and transform for training data
                    encoder = LabelEncoder()
                    df_processed[feature] = encoder.fit_transform(df_processed[feature].astype(str))
                    self.label_encoders[feature] = encoder
                else:
                    # Transform only for test data
                    if feature in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(df_processed[feature].astype(str))
                        known_values = set(self.label_encoders[feature].classes_)
                        
                        # Replace unseen categories with most frequent class
                        if unique_values - known_values:
                            most_frequent = self.label_encoders[feature].classes_[0]
                            df_processed[feature] = df_processed[feature].astype(str).apply(
                                lambda x: most_frequent if x not in known_values else x
                            )
                        
                        df_processed[feature] = self.label_encoders[feature].transform(df_processed[feature].astype(str))
        
        # Handle numerical features with scaling
        available_numerical = [col for col in self.numerical_features if col in df_processed.columns]
        
        if available_numerical:
            if is_training:
                # Fit and transform for training data
                df_processed[available_numerical] = self.scaler.fit_transform(df_processed[available_numerical])
            else:
                # Transform only for test data
                df_processed[available_numerical] = self.scaler.transform(df_processed[available_numerical])
        
        return df_processed
    
    def train_with_kfold(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
        """
        Train model using K-fold cross-validation
        
        Args:
            X: Feature matrix
            y: Target variable
            n_splits: Number of folds for cross-validation
            
        Returns:
            Dictionary with cross-validation results
        """
        # Initialize model with reasonable defaults
        model = XGBRegressor(
            random_state=self.random_state,
            n_estimators=100,  # Start with smaller value for CV
            max_depth=6,
            learning_rate=0.1,
            verbosity=0
        )
        
        # Perform K-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Use negative RMSE as scoring (scikit-learn convention)
        cv_scores = cross_val_score(
            model, X, y, 
            cv=kf, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        # Convert back to positive RMSE
        rmse_scores = -cv_scores
        
        self.cv_scores = rmse_scores
        
        results = {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'individual_scores': rmse_scores.tolist()
        }
        
        return results
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 3) -> Dict[str, Any]:
        """
        Tune hyperparameters using GridSearchCV
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of folds for hyperparameter tuning
            
        Returns:
            Dictionary with best parameters and results
        """
        # Define parameter grid based on your successful notebook parameters
        param_grid = {
            'n_estimators': [400, 600, 800],
            'max_depth': [8, 10, 12],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Initialize base model
        base_model = XGBRegressor(
            random_state=self.random_state,
            verbosity=0
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("Starting hyperparameter tuning...")
        grid_search.fit(X, y)
        print("Hyperparameter tuning completed!")
        
        self.best_params = grid_search.best_params_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_rmse': -grid_search.best_score_,
            'all_results': grid_search.cv_results_
        }
        
        return results
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train the final model with best parameters
        
        Args:
            X: Feature matrix
            y: Target variable
            params: Model parameters (if None, uses tuned parameters or defaults)
            
        Returns:
            Dictionary with training results
        """
        # Use provided params, or best params from tuning, or your successful notebook params
        if params is None:
            if self.best_params is not None:
                model_params = self.best_params.copy()
            else:
                # Your successful parameters from notebook
                model_params = {
                    'n_estimators': 800,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'subsample': 0.9
                }
        else:
            model_params = params.copy()
        
        # Add fixed parameters
        model_params.update({
            'random_state': self.random_state,
            'verbosity': 0
        })
        
        # Initialize and train model
        self.model = XGBRegressor(**model_params)
        
        print("Training final model...")
        self.model.fit(X, y)
        
        # Calculate training RMSE
        y_pred = self.model.predict(X)
        training_rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        results = {
            'model_params': model_params,
            'training_rmse': training_rmse,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        print(f"Final model training completed! Training RMSE: {training_rmse:.2f}")
        
        return results
    
    def save_models(self, model_dir: str = "models/") -> Dict[str, str]:
        """
        Save trained model and preprocessors
        
        Args:
            model_dir: Directory to save models
            
        Returns:
            Dictionary with saved file paths
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save trained model
        if self.model is not None:
            model_path = os.path.join(model_dir, "xgboost_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            saved_files['model'] = model_path
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        saved_files['scaler'] = scaler_path
        
        # Save label encoders
        encoder_path = os.path.join(model_dir, "encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        saved_files['encoders'] = encoder_path
        
        print(f"Models saved successfully to {model_dir}")
        return saved_files
    
    def load_models(self, model_dir: str = "models/") -> bool:
        """
        Load trained model and preprocessors
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            Boolean indicating successful loading
        """
        import os
        
        try:
            # Load trained model
            model_path = os.path.join(model_dir, "xgboost_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load label encoders
            encoder_path = os.path.join(model_dir, "encoder.pkl")
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
            
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train_final_model() or load_models() first.")
        
        # Prepare features (scaling and encoding)
        X_processed = self.prepare_features_for_training(X, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Returns:
            DataFrame with features and their importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_final_model() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.model.get_booster().feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def complete_training_pipeline(train_data: pd.DataFrame, target_column: str = 'Sales') -> ModelTrainer:
    """
    Complete training pipeline - convenience function
    
    Args:
        train_data: Training dataset with features and target
        target_column: Name of target column
        
    Returns:
        Trained ModelTrainer instance
    """
    # Separate features and target
    X = train_data.drop(columns=[target_column])
    y = train_data[target_column]
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare features
    print("Preparing features...")
    X_processed = trainer.prepare_features_for_training(X, is_training=True)
    
    # Train with cross-validation
    print("Performing cross-validation...")
    cv_results = trainer.train_with_kfold(X_processed, y)
    print(f"Cross-validation RMSE: {cv_results['mean_rmse']:.2f} ± {cv_results['std_rmse']:.2f}")
    
    # Train final model (using your successful parameters)
    print("Training final model...")
    final_results = trainer.train_final_model(X_processed, y)
    
    # Save models
    print("Saving models...")
    trainer.save_models()
    
    return trainer


if __name__ == "__main__":
    # Example usage
    print("Model Training Module - Rossmann Sales Forecasting")
    print("This module provides complete model training functionality.")
    print("\nKey functions:")
    print("- complete_training_pipeline(): Full training workflow")
    print("- ModelTrainer class: Detailed control over training process")
    print("\nReady for integration with your preprocessing and feature engineering modules!")