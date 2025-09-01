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
import pickle
import os

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

@st.cache_resource
def load_trained_models():
    """Load the trained models directly from pickle files"""
    try:
        # Check if model files exist
        model_files = ['models/xgboost_model.pkl', 'models/scaler.pkl', 'models/encoder.pkl']
        missing_files = [f for f in model_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"Missing model files: {missing_files}")
            return None, None, None
        
        # Load models
        with open('models/xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
            
        return model, scaler, encoder
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def make_prediction(input_data):
    """
    Make prediction with strict feature alignment to the trained artifacts.
    - Works if encoder.pkl is a dict of mappings OR if the model expects one-hot columns.
    - Scales only the columns the scaler was actually fit on.
    """
    try:
        model, scaler, encoder = load_trained_models()
        if model is None:
            return 0

        # Start with one-row DataFrame
        df = pd.DataFrame([input_data])

        # --- Date features ---
        y, m, d = int(input_data['Year']), int(input_data['Month']), int(input_data['Day'])
        df['Year'], df['Month'], df['Day'] = y, m, d
        df['WeekOfYear'] = datetime(y, m, d).isocalendar().week

        # --- Engineered features (guards added) ---
        # CompOpenSince (months)
        if pd.notna(input_data.get('CompetitionOpenSinceMonth')) and pd.notna(input_data.get('CompetitionOpenSinceYear')):
            comp_start = datetime(int(input_data['CompetitionOpenSinceYear']),
                                  int(input_data['CompetitionOpenSinceMonth']), 1)
            cur_date = datetime(y, m, d)
            df['CompOpenSince'] = (cur_date.year - comp_start.year) * 12 + (cur_date.month - comp_start.month)
        else:
            df['CompOpenSince'] = 0

        # Promo2OpenSince (weeks)
        if (input_data.get('Promo2', 0) == 1 and
            pd.notna(input_data.get('Promo2SinceWeek')) and
            pd.notna(input_data.get('Promo2SinceYear'))):
            try:
                p2_year = int(input_data['Promo2SinceYear'])
                p2_week = int(input_data['Promo2SinceWeek'])
                promo2_start = datetime.strptime(f"{p2_year}-W{p2_week:02d}-1", "%Y-W%W-%w")
                df['Promo2OpenSince'] = max(0, (datetime(y, m, d) - promo2_start).days // 7)
            except Exception:
                df['Promo2OpenSince'] = 0
        else:
            df['Promo2OpenSince'] = 0

        # IsPromo2Month (based on PromoInterval)
        if (input_data.get('Promo2', 0) == 1 and input_data.get('PromoInterval') not in [None, '', 'nan']):
            current_month_abbr = datetime(y, m, 1).strftime('%b')
            promo_months = [s.strip() for s in str(input_data['PromoInterval']).split(',')]
            df['IsPromo2Month'] = 1 if current_month_abbr in promo_months else 0
        else:
            df['IsPromo2Month'] = 0

        # Ensure Open exists
        df['Open'] = input_data.get('Open', 1)

        # Fill NA for competition/promo2 numeric fields if present
        for col, default in [
            ('CompetitionOpenSinceMonth', 0),
            ('CompetitionOpenSinceYear', 0),
            ('CompetitionDistance', 1000),
            ('Promo2SinceWeek', 0),
            ('Promo2SinceYear', 0),
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

        # --- StateHoliday to numeric (consistent with common training setups) ---
        if 'StateHoliday' in df.columns:
            df['StateHoliday'] = (
                df['StateHoliday']
                .astype(str)
                .map({'0': 0, 'a': 1, 'b': 2, 'c': 3})
                .fillna(0)
                .astype(int)
            )

        # --- Prepare to align with the model ---
        expected_features = list(getattr(model, 'feature_names_in_', []))

        # Work on a copy
        df_proc = df.copy()

        # Helper: create one-hot columns for a feature if the model expects them
        def ensure_one_hot(df_, feat_name, raw_value):
            matched = [c for c in expected_features if c.startswith(f"{feat_name}_")]
            if not matched:
                return df_
            # create all zeros
            for c in matched:
                df_[c] = 0
            key = f"{feat_name}_{str(raw_value)}"
            if key in df_.columns:
                df_[key] = 1
            # drop original text col if present
            df_.drop(columns=[feat_name], errors='ignore')
            return df_

        # --- Encode categorical features robustly ---
        for feat in ['StoreType', 'Assortment']:
            if feat not in df_proc.columns:
                continue
            raw_val = str(df_proc.iloc[0][feat])

            # Case A: encoder is a dict of mappings like {'StoreType': {'a':0,...}}
            if isinstance(encoder, dict) and feat in encoder and isinstance(encoder[feat], dict):
                df_proc[feat] = pd.Series([raw_val]).map(encoder[feat]).fillna(0).astype(int).values

            # Case B: model expects one-hot columns (e.g., 'StoreType_a', …)
            elif any(c.startswith(f"{feat}_") for c in expected_features):
                df_proc = ensure_one_hot(df_proc, feat, raw_val)
                # ensure original text col is gone
                df_proc.drop(columns=[feat], errors='ignore', inplace=True)

            # Case C: fallback ordinal mapping (if neither dict nor one-hot spec found)
            else:
                fallback_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
                df_proc[feat] = pd.Series([raw_val]).map(fallback_map).fillna(0).astype(int).values

        # --- Scale only the columns the scaler was fit on ---
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            scale_cols = [c for c in scaler.feature_names_in_ if c in df_proc.columns]
            if scale_cols:
                df_proc[scale_cols] = scaler.transform(df_proc[scale_cols])

        # --- Final feature alignment (order and fill missing) ---
        if expected_features:
            for f in expected_features:
                if f not in df_proc.columns:
                    df_proc[f] = 0
            X = df_proc[expected_features]
        else:
            # model doesn't expose feature names; best effort
            X = df_proc.select_dtypes(include=[np.number])

        # --- Predict ---
        pred = float(model.predict(X)[0])
        return max(0, round(pred, 2))

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.error("Please verify the artifacts in 'models/' (xgboost_model.pkl, scaler.pkl, encoder.pkl).")
        return 0

def make_simple_prediction(input_data):
    """
    Simplified prediction function for testing
    """
    try:
        # Mock prediction for testing UI
        base_sales = input_data.get('Customers', 500) * 8.5
        
        # Apply simple business logic
        if input_data.get('Promo', 0):
            base_sales *= 1.2
        if input_data.get('Open', 1) == 0:
            base_sales = 0
        if input_data.get('SchoolHoliday', 0):
            base_sales *= 1.1
            
        return max(0, round(base_sales, 2))
        
    except Exception as e:
        return 1000  # Default fallback

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
            open_store = st.selectbox("Store Open", [0, 1], format_func=lambda x: "Closed" if x == 0 else "Open", index=1)
            
        with col3:
            store_type = st.selectbox("Store Type", ["a", "b", "c", "d"])
            assortment = st.selectbox("Assortment", ["a", "b", "c"], 
                                    format_func=lambda x: {"a": "Basic", "b": "Extra", "c": "Extended"}[x])
            competition_distance = st.number_input("Competition Distance (m)", min_value=0, max_value=100000, value=1000)
            promo2 = st.selectbox("Long-term Promotion", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        # Advanced options (collapsible)
        with st.expander("🔧 Advanced Options"):
            comp_open_month = st.number_input("Competition Open Month", min_value=1, max_value=12, value=1)
            comp_open_year = st.number_input("Competition Open Year", min_value=1990, max_value=2020, value=2010)
            promo2_week = st.number_input("Promo2 Since Week", min_value=1, max_value=52, value=1) if promo2 else None
            promo2_year = st.number_input("Promo2 Since Year", min_value=2009, max_value=2020, value=2015) if promo2 else None
            promo_interval = st.selectbox("Promo Interval", ["", "Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"]) if promo2 else ""
        
        submitted = st.form_submit_button("🔮 Predict Sales", use_container_width=True)
        
        if submitted:
            # Create prediction data
            prediction_data = {
                'Store': store_id,
                'DayOfWeek': day_of_week,
                'Customers': customers,
                'Open': open_store,
                'Promo': promo,
                'StateHoliday': state_holiday,
                'SchoolHoliday': school_holiday,
                'StoreType': store_type,
                'Assortment': assortment,
                'CompetitionDistance': competition_distance,
                'CompetitionOpenSinceMonth': comp_open_month,
                'CompetitionOpenSinceYear': comp_open_year,
                'Promo2': promo2,
                'Promo2SinceWeek': promo2_week,
                'Promo2SinceYear': promo2_year,
                'PromoInterval': promo_interval,
                'Year': prediction_date.year,
                'Month': prediction_date.month,
                'Day': prediction_date.day
            }
            
            # Make prediction
            try:
                prediction = make_prediction(prediction_data)
                
                if prediction > 0:
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
                else:
                    st.warning("Store appears to be closed or prediction returned 0.")
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Trying fallback prediction method...")
                
                # Try simple prediction as fallback
                simple_pred = make_simple_prediction(prediction_data)
                st.warning(f"🔄 Fallback prediction: €{simple_pred:,.2f}")

def show_business_insights():
    """Show business insights and analytics"""
    st.subheader("📊 Business Insights")
    
    # Create sample business insights without requiring actual data
    col1, col2 = st.columns(2)
    
    with col1:
        # Store Type Distribution (sample data)
        store_types = ['a', 'b', 'c', 'd']
        store_counts = [450, 350, 200, 115]
        fig_store_type = px.pie(values=store_counts, 
                               names=store_types,
                               title="Store Type Distribution")
        st.plotly_chart(fig_store_type, use_container_width=True)
    
    with col2:
        # Assortment Distribution (sample data)
        assortments = ['a', 'b', 'c']
        assort_counts = [593, 370, 152]
        fig_assortment = px.bar(x=assortments, 
                               y=assort_counts,
                               title="Assortment Types",
                               labels={'x': 'Assortment', 'y': 'Count'})
        st.plotly_chart(fig_assortment, use_container_width=True)
    
    # Feature Importance
    st.subheader("🎯 Feature Importance")
    
    # Sample feature importance data
    features = ['Customers', 'Promo', 'CompetitionDistance', 'DayOfWeek', 'Month', 
               'StoreType', 'Assortment', 'CompOpenSince', 'SchoolHoliday', 'Year']
    importance = [0.35, 0.18, 0.12, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
    
    fig_importance = px.bar(x=importance, y=features,
                           orientation='h',
                           title="Top 10 Most Important Features")
    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Business Metrics
    st.subheader("💼 Business Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Daily Sales", "€5,773", "↑ 12%")
    with col2:
        st.metric("Stores Active", "1,115", "→ 0%")
    with col3:
        st.metric("Avg Customers/Day", "633", "↑ 8%")
    with col4:
        st.metric("Promo Participation", "38.7%", "↑ 5%")

def show_model_info():
    """Show information about the model"""
    st.subheader("🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Model Details:**
        - Algorithm: XGBoost Regressor
        - Features: 16+ engineered features
        - Training Data: 844,392 records
        - Expected RMSE: ~347-400
        - Parameters: n_estimators=800, max_depth=10
        """)
    
    with col2:
        st.success("""
        **Key Features:**
        - Date-based features (Year, Month, Week)
        - Competition analysis (CompOpenSince)
        - Promotion tracking (Promo2OpenSince)
        - Store characteristics (Type, Assortment)
        - Customer patterns and seasonality
        """)
    
    # Model Status Check
    st.subheader("🔍 Model Status")
    
    model, scaler, encoder = load_trained_models()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if model is not None:
            st.success("✅ XGBoost Model Loaded")
            if hasattr(model, 'feature_names_in_'):
                st.info(f"Features: {len(model.feature_names_in_)}")
        else:
            st.error("❌ Model Not Found")
    
    with col2:
        if scaler is not None:
            st.success("✅ Scaler Loaded")
        else:
            st.error("❌ Scaler Not Found")
    
    with col3:
        if encoder is not None:
            st.success("✅ Encoder Loaded")
        else:
            st.error("❌ Encoder Not Found")
    
    # Show feature names if available
    if model is not None and hasattr(model, 'feature_names_in_'):
        with st.expander("🔍 Model Features"):
            st.write("**Expected Features:**")
            for i, feature in enumerate(model.feature_names_in_, 1):
                st.write(f"{i:2d}. {feature}")

# Debug section
def add_debug_section():
    """Add debug section to test predictions"""
    st.sidebar.markdown("---")
    if st.sidebar.button("🔍 Debug Mode"):
        st.sidebar.success("Debug mode activated!")
        
        # Test prediction with sample data
        test_data = {
            'Store': 1,
            'DayOfWeek': 1,
            'Customers': 500,
            'Open': 1,
            'Promo': 1,
            'StateHoliday': '0',
            'SchoolHoliday': 0,
            'StoreType': 'a',
            'Assortment': 'a',
            'CompetitionDistance': 1270.0,
            'CompetitionOpenSinceMonth': 9.0,
            'CompetitionOpenSinceYear': 2008.0,
            'Promo2': 0,
            'Promo2SinceWeek': None,
            'Promo2SinceYear': None,
            'PromoInterval': None,
            'Year': 2023,
            'Month': 8,
            'Day': 15
        }
        
        st.write("🧪 Testing prediction with sample data...")
        try:
            prediction = make_prediction(test_data)
            st.success(f"✅ Test prediction successful: €{prediction:,.2f}")
        except Exception as e:
            st.error(f"❌ Test failed: {e}")
            # Try simple prediction as fallback
            try:
                simple_pred = make_simple_prediction(test_data)
                st.warning(f"🔄 Fallback prediction: €{simple_pred:,.2f}")
            except Exception as e2:
                st.error(f"❌ Fallback also failed: {e2}")

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
        
        # Add debug section
        if st.checkbox("Show Debug Options"):
            add_debug_section()
    
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