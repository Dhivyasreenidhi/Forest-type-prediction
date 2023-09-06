import streamlit as st
import numpy as np
import pandas as pd
import tensorflow 
import keras
from sklearn.preprocessing import StandardScaler

# Load your pre-trained model
#modeling
model=tensorflow.keras.Sequential()
model.add(keras.layers.Dense(14,input_dim=14,activation='relu'))
model.add(keras.layers.Dense(100,input_dim=200,activation='relu'))
model.add(keras.layers.Dense(300,input_dim=400,activation='relu'))
model.add(keras.layers.Dense(400,input_dim=500,activation='relu'))
model.add(keras.layers.Dense(500,input_dim=600,activation='relu'))
model.add(keras.layers.Dense(600,input_dim=700,activation='relu'))
model.add(keras.layers.Dense(700,input_dim=800,activation='relu'))
model.add(keras.layers.Dense(800,input_dim=900,activation='relu'))
model.add(keras.layers.Dense(8,input_dim=200,activation = "softmax"))


# Load pre-trained weights (modify the path accordingly)
model.load_weights('model1.h5')

# Streamlit UI
st.title("Forest Cover Type Prediction")

st.sidebar.header("User Input")

# Add input fields for user to enter feature values
Elevation = st.sidebar.number_input("Elevation", min_value= 1000, max_value=2000, value=1000)
Aspect = st.sidebar.number_input("Aspect", min_value=0, max_value=1000, value=0)
Slope = st.sidebar.number_input("Slope", min_value=0, max_value=100, value=0)
Horizontal_Distance_To_Hydrology = st.sidebar.number_input("Horizontal_Distance_To_Hydrology", min_value=0, max_value=10000, value=0)
Vertical_Distance_To_Hydrology = st.sidebar.number_input("Vertical_Distance_To_Hydrology", min_value=0, max_value=1000, value=0)
Horizontal_Distance_To_Roadways = st.sidebar.number_input("Horizontal_Distance_To_Roadways", min_value=-100, max_value=1000, value=0)
Hillshade_9am = st.sidebar.number_input("Hillshade_9am", min_value=0, max_value=10000, value=0)
Hillshade_Noon = st.sidebar.number_input("Hillshade_Noon", min_value=0, max_value=1000, value=0)
Hillshade_3pm = st.sidebar.number_input("Hillshade_3pm", min_value=0, max_value=1000, value=0)
Horizontal_Distance_To_Fire_Points = st.sidebar.number_input("Horizontal_Distance_To_Fire_Points", min_value=0, max_value=10000, value=0)
Wilderness_Area1 = st.sidebar.number_input("Wilderness_Area1", min_value=0, max_value=1, value=0)
Wilderness_Area2 = st.sidebar.number_input("Wilderness_Area2", min_value=0, max_value=1, value=0)
Wilderness_Area3 = st.sidebar.number_input("Wilderness_Area3", min_value=0, max_value=1, value=0)
Wilderness_Area4 = st.sidebar.number_input("Wilderness_Area4", min_value=0, max_value=1, value=0)

# Make predictions when the user clicks the "Predict" button
if st.sidebar.button("Predict"):
    # Create a new data point from user inputs
    user_data = np.array([[Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Hillshade_9am,Hillshade_Noon,Hillshade_3pm,Horizontal_Distance_To_Fire_Points,Wilderness_Area1,Wilderness_Area2,Wilderness_Area3,Wilderness_Area4]])
    # Create a StandardScaler object

    
    # Perform data normalization
    scaler = StandardScaler()
    # Fit the scaler to your training data (modify with your actual training data)
    # scaler.fit(training_data)
    # Transform user input data
    user_data_scaled = scaler.fit_transform(user_data)

    # Make predictions using the loaded model
    prediction = model.predict(user_data_scaled)

    # Display the prediction
    if prediction[0][0] == 1:
        st.write("It is Spruce/Fur forest")
    if prediction[0][0] == 2:
        st.write("It is Lodgepole Pine forest")
    if prediction[0][0] == 3:
        st.write("It is Ponderosa Pine forest")
    if prediction[0][0] == 4:
        st.write("It is Cottonwood/Willow forest")
    if prediction[0][0] == 5:
        st.write("It is Aspen forest")
    if prediction[0][0] == 6:
        st.write("It is Douglas-fir forest")
    else:
        st.write("It is Krummholz forest")

# Optionally, you can add some information or instructions
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("1. Enter the values for the features.")
st.sidebar.markdown("2. Click the 'Predict' button to see the prediction.")

# You can also add some additional information or description about your model or dataset
st.write("This is a Streamlit app for Forest Cover Type  prediction.")

