import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the columns
cols = [
    'potential', 'value_eur', 'wage_eur', 'age',
    'international_reputation', 'shooting', 'passing', 'dribbling',
    'physic', 'attacking_crossing', 'attacking_short_passing',
    'movement_reactions', 'mentality', 'technical_skills', 'goal_keeping'
]

# Load the model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Prediction function
def predict(model, features):
    df = pd.DataFrame([features], columns=features.keys())
    prediction = model.predict(df)
    return prediction[0]

# Main function to set up the Streamlit app
def main():
    st.title("FIFA Player Prediction")
    html_temp = """
    <div style="background-color:#025246; padding:10px;">
    <h2 style="color:white; text-align:center;">FIFA Player Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create input fields
    features = {}
    for feature in cols:
        features[feature] = st.number_input(feature, value=0.0)

    model_path = 'ensemble_model.joblib'
    model = load_model(model_path)

    if st.button("Predict") and model is not None:
        prediction = predict(model, features)
        st.write(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()
