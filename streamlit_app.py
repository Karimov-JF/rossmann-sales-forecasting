"""
Rossmann Sales Forecasting Dashboard
Interactive Streamlit application for sales prediction and business insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import sys
import os

# Add src to path for imports
sys.path.append('src')

# Import your modules
try:
    from data_preprocessing import preprocess_pipeline
    from feature_engineering import feature_engineering_pipeline
    from model_training import ModelTrainer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure you're running this from the main project directory")
    st.stop()

# Page config
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample data for the demo"""
    try:
        _, test_data, store_data = preprocess_pipeline()
        test_features = feature_engineering_pipeline(test_data)
        return test_features.head(1000), store_data  # Return first 1000 rows for demo
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def load_trained_model():
    """Load the trained model"""
    try:
        trainer = ModelTrainer()
        if trainer.load_models():
            return trainer
        else:
            st.error("Could not load trained models. Make sure models/ directory contains the trained models.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_prediction_form():
    """Create the prediction input form"""
    st.subheader("🎯 Sales Prediction")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
            day_of_week = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7], 
                                     format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x-1])
            customers = st.number_input("Expected Customers", min_value=0, max_value=5000, value=500)
            promo = st.selectbox("Promotion Active", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
        with col2:
            prediction_date = st.date_input("Prediction Date", value=date(2015, 8, 1))
            school_holiday = st.selectbox("School Holiday", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            state_holiday = st.selectbox("State Holiday", ["0", "a", "b", "c"], 
                                       format_func=lambda x: {"0": "None", "a": "Public", "b": "Easter", "c": "Christmas"}[x])
            
        with col3:
            store_type = st.selectbox("Store Type", ["a", "b", "c", "d"])
            assortment = st.selectbox("Assortment", ["a", "b", "c"], 
                                    format_func=lambda x: {"a": "Basic", "b": "Extra", "c": "Extended"}[x])
            competition_distance = st.number_input("Competition Distance (m)", min_value=0, max_value=100000, value=1000)
            promo2 = st.selectbox("Long-term Promotion", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        submitted = st.form_submit_button("🔮 Predict Sales", use_container_width=True)
        
        if submitted:
            # Create prediction data
            prediction_data = create_prediction_dataframe(
                store_id, day_of_week, customers, promo, prediction_date, 
                school_holiday, state_holiday, store_type, assortment, 
                competition_distance, promo2
            )
            
            # Make prediction
            trainer = load_trained_model()
            if trainer:
                try:
                    prediction = trainer.predict(prediction_data)[0]
                    
                    # Display prediction with styling
                    st.markdown(f"""
                    <div class="prediction-result" style="background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;">
                        💰 Predicted Sales: €{prediction:,.2f}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional insights
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Daily Revenue", f"€{prediction:,.0f}")
                    with col2:
                        revenue_per_customer = prediction / max(customers, 1)
                        st.metric("Revenue per Customer", f"€{revenue_per_customer:.2f}")
                    with col3:
                        monthly_estimate = prediction * 30
                        st.metric("Monthly Estimate", f"€{monthly_estimate:,.0f}")
                        
                except Exception as e:
                    st.error(f"Prediction error: {e}")

def create_prediction_dataframe(store_id, day_of_week, customers, promo, prediction_date, 
                               school_holiday, state_holiday, store_type, assortment, 
                               competition_distance, promo2):
    """Create dataframe for prediction"""
    
    # Extract date features
    year = prediction_date.year
    month = prediction_date.month
    day = prediction_date.day
    week_of_year = prediction_date.isocalendar()[1]
    
    # Create the prediction dataframe with all required features
    data = {
        'Store': [store_id],
        'DayOfWeek': [day_of_week],
        'Customers': [customers],
        'Promo': [promo],
        'SchoolHoliday': [school_holiday],
        'StoreType': [store_type],
        'Assortment': [assortment],
        'CompetitionDistance': [competition_distance],
        'CompetitionOpenSinceMonth': [1],  # Default values
        'CompetitionOpenSinceYear': [2010],
        'Promo2': [promo2],
        'StateHoliday': [state_holiday],
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'WeekOfYear': [week_of_year],
        'CompOpenSince': [60],  # Default: 5 years
        'Promo2OpenSince': [0 if promo2 == 0 else 52],  # Default: 1 year if active
        'IsPromo2Month': [1 if promo2 == 1 else 0]
    }
    
    return pd.DataFrame(data)

def show_business_insights():
    """Show business insights and analytics"""
    st.subheader("📊 Business Insights")
    
    # Load sample data for insights
    sample_data, store_data = load_sample_data()
    
    if sample_data is not None and store_data is not None:
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Store Type Distribution
            store_type_counts = store_data['StoreType'].value_counts()
            fig_store_type = px.pie(values=store_type_counts.values, 
                                   names=store_type_counts.index,
                                   title="Store Type Distribution")
            st.plotly_chart(fig_store_type, use_container_width=True)
        
        with col2:
            # Assortment Distribution
            assortment_counts = store_data['Assortment'].value_counts()
            fig_assortment = px.bar(x=assortment_counts.index, 
                                   y=assortment_counts.values,
                                   title="Assortment Types",
                                   labels={'x': 'Assortment', 'y': 'Count'})
            st.plotly_chart(fig_assortment, use_container_width=True)
        
        # Feature Importance (if model is available)
        trainer = load_trained_model()
        if trainer and hasattr(trainer, 'model') and trainer.model is not None:
            st.subheader("🎯 Feature Importance")
            
            try:
                feature_importance = trainer.get_feature_importance()
                
                fig_importance = px.bar(feature_importance.head(10), 
                                       x='importance', 
                                       y='feature',
                                       orientation='h',
                                       title="Top 10 Most Important Features")
                fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.info(f"Feature importance not available: {e}")
        
        # Competition Analysis
        st.subheader("🏪 Competition Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average competition distance by store type
            comp_by_type = store_data.groupby('StoreType')['CompetitionDistance'].mean().fillna(0)
            fig_comp = px.bar(x=comp_by_type.index, 
                            y=comp_by_type.values,
                            title="Avg Competition Distance by Store Type",
                            labels={'x': 'Store Type', 'y': 'Distance (m)'})
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with col2:
            # Promo2 participation by assortment
            if 'Promo2' in store_data.columns:
                promo2_by_assort = store_data.groupby('Assortment')['Promo2'].mean() * 100
                fig_promo2 = px.bar(x=promo2_by_assort.index, 
                                   y=promo2_by_assort.values,
                                   title="Promo2 Participation by Assortment (%)",
                                   labels={'x': 'Assortment', 'y': 'Participation %'})
                st.plotly_chart(fig_promo2, use_container_width=True)

def show_model_info():
    """Show information about the model"""
    st.subheader("🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Model Details:**
        - Algorithm: XGBoost Regressor
        - Features: 16 engineered features
        - Training Data: 844,392 records
        - Performance: RMSE ~347-400
        """)
    
    with col2:
        st.success("""
        **Key Features:**
        - Date-based features (Year, Month, Week)
        - Competition analysis (CompOpenSince)
        - Promotion tracking (Promo2OpenSince)
        - Store characteristics (Type, Assortment)
        """)
    
    # Model performance metrics (if available)
    trainer = load_trained_model()
    if trainer:
        st.subheader("📈 Model Performance")
        
        # Create some sample predictions for demonstration
        sample_data, _ = load_sample_data()
        if sample_data is not None:
            try:
                # Make predictions on sample
                sample_for_pred = sample_data.head(100).drop(columns=['Sales'], errors='ignore')
                predictions = trainer.predict(sample_for_pred)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Prediction", f"€{np.mean(predictions):,.0f}")
                with col2:
                    st.metric("Min Prediction", f"€{np.min(predictions):,.0f}")
                with col3:
                    st.metric("Max Prediction", f"€{np.max(predictions):,.0f}")
                
                # Distribution of predictions
                fig_dist = px.histogram(x=predictions, 
                                       title="Distribution of Sample Predictions",
                                       labels={'x': 'Predicted Sales (€)', 'y': 'Count'})
                st.plotly_chart(fig_dist, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate sample predictions: {e}")

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏪 Rossmann Sales Forecasting</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Choose a page:", [
            "🎯 Sales Prediction", 
            "📊 Business Insights", 
            "🤖 Model Information"
        ])
        
        st.markdown("---")
        st.info("""
        **About This App:**
        
        AI-powered sales forecasting system for Rossmann drugstores. 
        
        Enter store parameters to get accurate sales predictions and business insights.
        """)
        
        st.markdown("---")
        st.markdown("**Built with:**")
        st.markdown("- 🐍 Python")
        st.markdown("- 🤖 XGBoost")
        st.markdown("- 📊 Streamlit")
        st.markdown("- 📈 Plotly")
    
    # Main content
    if page == "🎯 Sales Prediction":
        create_prediction_form()
    elif page == "📊 Business Insights":
        show_business_insights()
    else:  # Model Information
        show_model_info()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🚀 Built by Karimov | AI-Powered Business Solutions</p>
        <p>Rossmann Sales Forecasting System v1.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()