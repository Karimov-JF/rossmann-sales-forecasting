# Rossmann Sales Forecasting

A complete machine learning pipeline for predicting daily sales of Rossmann stores using XGBoost, featuring an interactive Streamlit dashboard for business insights and predictions.

## Features

- **Complete ML Pipeline**: Modular architecture with data preprocessing, feature engineering, and model training
- **Feature Engineering**: Custom features including competition analysis and promotional timing
- **XGBoost Model**: Optimized regression model with proven parameters (RMSE ~347-400)
- **Interactive Dashboard**: Streamlit web application for predictions and business analytics
- **Production Ready**: Scalable code structure with proper error handling and model persistence

## Dataset

The project uses the Rossmann Store Sales dataset from Kaggle, containing:
- **Training Data**: 1,017,209 records with daily sales data
- **Store Data**: 1,115 stores with characteristics (type, assortment, competition info)
- **Test Data**: 41,088 records for evaluation

## Project Structure

```
rossmann-sales-forecasting/
├── data/                          # Dataset files
│   ├── train.csv
│   ├── test.csv
│   └── store.csv
├── src/                           # Source code modules
│   ├── data_preprocessing.py      # Data loading and cleaning
│   ├── feature_engineering.py    # Custom feature creation
│   └── model_training.py         # XGBoost training pipeline
├── models/                        # Trained model artifacts
│   ├── xgboost_model.pkl         # Trained XGBoost model
│   ├── scaler.pkl                # Feature scaler
│   └── encoder.pkl               # Categorical encoders
├── notebooks/                     # Jupyter notebooks for exploration
├── streamlit_app.py              # Interactive web dashboard
├── train_model.py                # Complete training orchestrator
├── test_pipeline.py              # Pipeline validation script
└── README.md                     # This file
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rossmann-sales-forecasting.git
cd rossmann-sales-forecasting
```

2. **Create virtual environment**
```bash
python -m venv rossmann-env
source rossmann-env/bin/activate  # On Windows: rossmann-env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly
```

4. **Download dataset**
- Download the Rossmann Store Sales dataset from Kaggle
- Place `train.csv`, `test.csv`, and `store.csv` in the `data/` directory

## Usage

### Training the Model

Run the complete training pipeline:
```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Create engineered features
- Train the XGBoost model
- Save trained models to `models/` directory

### Running the Dashboard

Launch the interactive Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

The dashboard provides:
- **Sales Prediction**: Interactive form for making sales forecasts
- **Business Insights**: Data visualizations and analytics
- **Model Information**: Performance metrics and feature importance

### Testing the Pipeline

Validate the entire pipeline:
```bash
python test_pipeline.py
```

## Model Details

### Algorithm
- **XGBoost Regressor** with optimized hyperparameters
- Cross-validation RMSE: ~347-400
- Training data: 844,392 records (filtered for open stores)

### Key Parameters
```python
{
    'n_estimators': 800,
    'max_depth': 10,
    'learning_rate': 0.1,
    'subsample': 0.9
}
```

### Feature Engineering

The model uses 16 engineered features:

**Date Features**:
- Year, Month, Day, WeekOfYear

**Store Features**:
- Store ID, DayOfWeek, Open status
- StoreType, Assortment, CompetitionDistance

**Promotional Features**:
- Promo (daily promotions)
- StateHoliday, SchoolHoliday

**Custom Engineered Features**:
- **CompOpenSince**: Months since competition opened
- **Promo2OpenSince**: Weeks since long-term promotion started  
- **IsPromo2Month**: Whether current month is in promotional interval

## Model Performance

- **Training RMSE**: ~347-400
- **Cross-validation**: 5-fold validation with consistent performance
- **Feature Importance**: Date and promotional features are most predictive

## Dashboard Features

### Sales Prediction
- Input store parameters (ID, date, customer count, promotions)
- Real-time sales forecasting
- Revenue metrics and insights

### Business Analytics
- Store type and assortment distributions
- Competition analysis by store characteristics
- Feature importance visualization
- Promotional effectiveness analysis

### Model Information
- Training details and performance metrics
- Feature engineering explanations
- Sample prediction distributions

## Technical Stack

- **Python 3.8+**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Plotly, Streamlit
- **Model Persistence**: Pickle

## Key Insights

1. **Promotional Impact**: Promotions significantly boost sales, with Promo2 showing long-term effects
2. **Temporal Patterns**: Strong weekly and monthly seasonality in sales
3. **Competition Effect**: Proximity to competitors affects sales patterns
4. **Store Characteristics**: Store type and assortment level are key predictors

## Future Enhancements

- **Real-time Data Integration**: Connect to live sales data streams
- **Advanced Models**: Experiment with ensemble methods and deep learning
- **Geographic Analysis**: Incorporate location-based features
- **Inventory Optimization**: Link sales forecasts to stock management
- **A/B Testing Framework**: Evaluate promotional strategies

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Rossmann for providing the dataset via Kaggle competition
- Kaggle community for insights and methodologies
- Streamlit team for the excellent visualization framework

## Contact

**Karimov Jamoliddin** - kjamoliddin887@gmail.com
**LinkedIn**: [www.linkedin.com/in/jamoliddin-karimov]

---

*This project demonstrates end-to-end machine learning capabilities including data engineering, feature development, model training, and production deployment suitable for enterprise sales forecasting applications.*