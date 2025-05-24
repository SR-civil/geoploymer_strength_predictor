import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page title and layout
st.set_page_config(
    page_title="Geopolymer Concrete Strength Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add title and description
st.title("Geopolymer Concrete Strength Predictor")
st.write("""
This app predicts the compressive strength of geopolymer concrete based on various mix design parameters
and curing conditions using different machine learning models.
""")

# Function to load models
@st.cache_resource
def load_models():
    models = {}
    model_files = [
        'models/lasso_model.pkl',
        'models/decision_tree_model.pkl', 
        'models/random_forest_model.pkl',
        'models/ada_boost_model.pkl',
        'models/svr_model.pkl',
        'models/knn_model.pkl'
    ]
    for model_file in model_files:
        try:
            with open(model_file, 'rb') as file:
                model_name = model_file.split('_model.pkl')[0].split('/')[-1]
                models[model_name] = pickle.load(file)
        except FileNotFoundError:
            st.error(f"Model file {model_file} not found!")
    
    # Load scaler
    try:
        with open('models/scaler.pkl', 'rb') as file:
            models['scaler'] = pickle.load(file)
    except FileNotFoundError:
        st.warning("Scaler file not found - proceeding without scaling")
        
    return models

# Load models
models = load_models()
print(models)
# Check if models were loaded successfully
if not models:
    st.error("Failed to load models. Please check if the model files exist in the current directory.")
    st.stop()

# Define sidebar inputs
st.sidebar.header("Input Parameters")

# Non-percentage inputs
days_testing = st.sidebar.selectbox(
    "Number of days (testing)", 
    options=[7, 28, 90], 
    index=1  # This sets 28 as the default (index 1)
)


curing_options = ["Oven (0)", "Outdoor (1)"]
curing_selection = st.sidebar.selectbox("Curing type", curing_options)
curing_type = "0" if "Oven" in curing_selection else "1"

st.sidebar.markdown("---")
st.sidebar.subheader("Mix Composition Percentages")
st.sidebar.info("💡 The sum of all percentages must equal 100%")

# Initialize session state for percentages if not exists
if 'percentages_initialized' not in st.session_state:
    st.session_state.fly_ash = 10.0
    st.session_state.ggbs = 8.5
    st.session_state.sodium_silicate = 2.25
    st.session_state.sodium_hydroxide = 2.25
    st.session_state.sand = 25.0
    st.session_state.coarse_aggregate = 50.0
    st.session_state.glass_waste = 2.0
    st.session_state.percentages_initialized = True

# Percentage inputs with session state
fly_ash = st.sidebar.number_input(
    "Fly ash (%)", 
    min_value=0.0, 
    max_value=15.0, 
    value=st.session_state.fly_ash,
    step=0.1,
    key="fly_ash_input"
)

ggbs = st.sidebar.number_input(
    "GGBS (%)", 
    min_value=0.0, 
    max_value=15.0, 
    value=st.session_state.ggbs,
    step=0.1,
    key="ggbs_input"
)

sodium_silicate = st.sidebar.number_input(
    "Sodium silicate (%)", 
    min_value=0.0, 
    max_value=7.0, 
    value=st.session_state.sodium_silicate,
    step=0.1,
    key="sodium_silicate_input"
)

sodium_hydroxide = st.sidebar.number_input(
    "Sodium hydroxide (%)", 
    min_value=2.0, 
    max_value=2.5, 
    value=st.session_state.sodium_hydroxide,
    step=0.1,
    key="sodium_hydroxide_input"
)

sand = st.sidebar.number_input(
    "Sand (%)", 
    min_value=25.0, 
    max_value=30.0, 
    value=st.session_state.sand,
    step=0.1,
    key="sand_input"
)

coarse_aggregate = st.sidebar.number_input(
    "Coarse aggregate (%)", 
    min_value=40.0, 
    max_value=55.0, 
    value=st.session_state.coarse_aggregate,
    step=0.1,
    key="coarse_aggregate_input"
)

glass_waste = st.sidebar.number_input(
    "Glass waste (%)", 
    min_value=0.0, 
    max_value=5.0, 
    value=st.session_state.glass_waste,
    step=0.1,
    key="glass_waste_input"
)

# Calculate total percentage
total_percentage = fly_ash + ggbs + sodium_silicate + sodium_hydroxide + sand + coarse_aggregate + glass_waste

# Display percentage sum status
if total_percentage == 100.0:
    st.sidebar.success(f"✅ Total: {total_percentage}%")
    percentage_valid = True
elif total_percentage < 100.0:
    st.sidebar.warning(f"⚠️ Total: {total_percentage}% (Need {100.0 - total_percentage:.1f}% more)")
    percentage_valid = False
else:
    st.sidebar.error(f"❌ Total: {total_percentage}% (Exceeds by {total_percentage - 100.0:.1f}%)")
    percentage_valid = False

# Auto-normalize button
if not percentage_valid:
    if st.sidebar.button("🔄 Auto-normalize to 100%"):
        if total_percentage > 0:
            # Normalize all percentages proportionally
            factor = 100.0 / total_percentage
            st.session_state.fly_ash = round(fly_ash * factor, 1)
            st.session_state.ggbs = round(ggbs * factor, 1)
            st.session_state.sodium_silicate = round(sodium_silicate * factor, 1)
            st.session_state.sodium_hydroxide = round(sodium_hydroxide * factor, 1)
            st.session_state.sand = round(sand * factor, 1)
            st.session_state.coarse_aggregate = round(coarse_aggregate * factor, 1)
            st.session_state.glass_waste = round(glass_waste * factor, 1)
            # st.experimental_rerun()

# Reset button
if st.sidebar.button("🔄 Reset to defaults"):
    st.session_state.fly_ash = 10.0
    st.session_state.ggbs = 8.5
    st.session_state.sodium_silicate = 2.25
    st.session_state.sodium_hydroxide = 2.25
    st.session_state.sand = 25.0
    st.session_state.coarse_aggregate = 50.0
    st.session_state.glass_waste = 2.0

    # st.experimental_rerun()

# Create input dataframe for prediction
def prepare_input_data():
    input_data = pd.DataFrame({
        'Number of days (testing)': [days_testing],
        'fly_ash_percentage': [fly_ash],
        'ggbs_percentage': [ggbs],
        'sodium_silicate_percentage': [sodium_silicate],
        'sodium_hydroxide_percentage': [sodium_hydroxide],
        'sand_percentage': [sand],
        'coarse_aggregate_percentage': [coarse_aggregate],
        'glass_waste_percentage': [glass_waste],
        'Curing_type': [curing_type],
    })
    
    return input_data

def prepare_input_array():
    """Convert input data to numpy array to avoid feature name warnings"""
    input_data = prepare_input_data()
    return input_data.values

# Make prediction when user clicks the button
predict_button = st.sidebar.button("🔮 Predict Strength", disabled=not percentage_valid, type="primary")

if predict_button and percentage_valid:
    input_data = prepare_input_data()
    # input_array = prepare_input_array()  # Use numpy array for predictions
    if input_data is not None and models["scaler"] is not None:
  # Scale the user input
        user_data_scaled = models["scaler"].transform(input_data)
    # Display input summary
    st.subheader("📋 Input Parameters Summary")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Testing Days", f"{days_testing}")
        st.metric("Fly Ash", f"{fly_ash}%")
        st.metric("GGBS", f"{ggbs}%")
        
    with col2:
        st.metric("Sodium Silicate", f"{sodium_silicate}%")
        st.metric("Sodium Hydroxide", f"{sodium_hydroxide}%")
        st.metric("Sand", f"{sand}%")
        
    with col3:
        st.metric("Coarse Aggregate", f"{coarse_aggregate}%")
        st.metric("Glass Waste", f"{glass_waste}%")
        st.metric("Curing Type", curing_selection)
    
    # Make predictions using all models
    st.subheader("🎯 Compressive Strength Predictions")
    
    results = {}
    for model_name, model in models.items():
        if model_name != 'scaler':
            try:
                # Use numpy array instead of DataFrame to avoid feature name warnings
                prediction = model.predict(user_data_scaled)
               
                results[model_name] = round(prediction[0], 2)
            except Exception as e:
                st.error(f"Error predicting with {model_name}: {e}")
                # If numpy array fails, try with DataFrame as fallback
                try:
                    prediction = model.predict(user_data_scaled)
                    results[model_name] = round(prediction, 2)
                    st.warning(f"Used DataFrame fallback for {model_name}")
                except Exception as e2:
                    st.error(f"Both prediction methods failed for {model_name}: {e2}")
    
    # Display predictions in a table
    if results:
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Predicted Compressive Strength (MPa)': list(results.values())
        })
        
        # Style the dataframe
        styled_df = results_df.style.format({'Predicted Compressive Strength (MPa)': '{:.2f}'})
        st.dataframe(styled_df, use_container_width=True)
        
        # Calculate average prediction
        avg_prediction = round(np.mean(list(results.values())), 2)
        std_prediction = round(np.std(list(results.values())), 2)
        
        # Display average with metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Strength", f"{avg_prediction} MPa")
        with col2:
            st.metric("Standard Deviation", f"{std_prediction} MPa")
        with col3:
            st.metric("Range", f"{min(results.values()):.1f} - {max(results.values()):.1f} MPa")
        
        # Visualize predictions
        st.subheader("📊 Prediction Comparison")
        chart_data = pd.DataFrame({
            'Model': list(results.keys()),
            'Strength (MPa)': list(results.values())
        })
        st.bar_chart(chart_data.set_index('Model'))
        
    else:
        st.error("Failed to make predictions with any model.")

elif predict_button and not percentage_valid:
    st.error("Please ensure all percentages sum to exactly 100% before predicting.")

# Add information about the models
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About the Models")
st.sidebar.info("""
This app uses five different machine learning models:
- Lasso Regression
- Decision Tree
- Random Forest  
- AdaBoost
- Support Vector Regression (SVR)
- K-Nearest Neighbors (KNN)

**Note:** All percentage inputs must sum to exactly 100% for the prediction to work.
""")

# Add model performance info if available
if models:
    st.sidebar.success(f"✅ {len([k for k in models.keys() if k != 'scaler'])} models loaded successfully")
else:
    st.sidebar.error("❌ No models loaded")