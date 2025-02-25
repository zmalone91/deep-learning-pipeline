import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Load your Keras model (.keras file)
MODEL_PATH = "models/best_keras_model.keras"
model = load_model(MODEL_PATH)

st.title("Local Keras Model Inference App (Iris Example)")

st.write("""
This Streamlit app supports:
1. **Single** prediction (manual feature entry).
2. **Batch** predictions (CSV upload).
""")

# --- SINGLE PREDICTION SECTION ---
st.header("Single Prediction")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal length", value=5.1, format="%.2f")
    sepal_width = st.number_input("Sepal width", value=3.5, format="%.2f")
with col2:
    petal_length = st.number_input("Petal length", value=1.4, format="%.2f")
    petal_width = st.number_input("Petal width", value=0.2, format="%.2f")

if st.button("Predict Single Example"):
    # Convert user input to a 2D array for model inference
    single_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=np.float32)
    
    # Run inference
    single_pred = model.predict(single_input)
    
    # If your final layer is something like Dense(3, activation='softmax'),
    # single_pred might be an array of probabilities. We can do argmax or show them directly.
    
    # Example: if it's a multi-class with shape (1,3):
    st.write("Raw Prediction Output:", single_pred.tolist())
    
    predicted_class = np.argmax(single_pred, axis=1)
    st.write("Predicted Class Index:", predicted_class[0])
    
    # If you have a class mapping, e.g., 0->Setosa, 1->Versicolor, 2->Virginica:
    class_names = ["Setosa", "Versicolor", "Virginica"]
    st.write("Predicted Class Name:", class_names[predicted_class[0]])

# --- BATCH PREDICTION SECTION ---
st.header("Batch Prediction via CSV Upload")

st.write("Upload a CSV with columns: sepal_length, sepal_width, petal_length, petal_width")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    required_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        # Prepare the batch input
        X = df[required_cols].values.astype(np.float32)
        
        # Predict
        batch_preds = model.predict(X)  # shape might be (N,3)
        
        # Suppose it's a softmax for 3 classes
        pred_class_indices = np.argmax(batch_preds, axis=1)
        
        # Add columns to df
        df["predicted_class"] = pred_class_indices
        
        # If you want probabilities for each class
        prob_cols = [f"prob_class_{i}" for i in range(batch_preds.shape[1])]
        prob_df = pd.DataFrame(batch_preds, columns=prob_cols)
        
        df = pd.concat([df, prob_df], axis=1)
        
        st.dataframe(df)
        
        # Let user download results
        csv_out = df.to_csv(index=False)
        st.download_button("Download Predictions as CSV", data=csv_out, file_name="predictions.csv")
