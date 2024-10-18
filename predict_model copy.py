import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pickle
import argparse
import tensorflow
from tensorflow.keras.models import load_model
import os
import joblib

def load_scaler(filepath):
    return joblib.load(filepath)

def load_preprocessing_objects(scaler_x_filepath, power_transformer_x_filepath, poly_filepath, scaler_y_filepath):
    with open(scaler_x_filepath, 'rb') as file:
        scaler_x = pickle.load(file)

    with open(power_transformer_x_filepath, 'rb') as file:
        power_transformer_x = pickle.load(file)

    with open(poly_filepath, 'rb') as file:
        poly = pickle.load(file)

    with open(scaler_y_filepath, 'rb') as file:
        scaler_y = pickle.load(file)

    return scaler_x, power_transformer_x, poly, scaler_y

def make_inference(model_path, scaler_x_path, power_transformer_x_path, poly_path, scaler_y_path, new_x):
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model

    scaler_x = joblib.load(scaler_x_path)
    power_transformer_x = joblib.load(power_transformer_x_path)
    poly = joblib.load(poly_path)
    scaler_y = joblib.load(scaler_y_path)

    new_x_normalized = scaler_x.transform(new_x.reshape(-1, 1))  # Reshape new_x
    new_x_transformed = power_transformer_x.transform(new_x_normalized)
    new_x_poly = poly.transform(new_x_transformed)
    new_x_poly = new_x_poly.reshape((new_x_poly.shape[0], 1, new_x_poly.shape[1]))  # Reshape for LSTM

    predicted_y_normalized = model.predict(new_x_poly)
    predicted_y = scaler_y.inverse_transform(predicted_y_normalized)
    return predicted_y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make predictions with an LSTM model")
    parser.add_argument('model_filepath', type=str, help='Path to the trained LSTM model')
    parser.add_argument('scaler_x_filepath', type=str, help='Path to the X scaler object')
    parser.add_argument('power_transformer_x_filepath', type=str, help='Path to the power transformer object')
    parser.add_argument('poly_filepath', type=str, help='Path to the polynomial features object')
    parser.add_argument('scaler_y_filepath', type=str, help='Path to the Y scaler object')
    parser.add_argument('new_x_value', type=float, help='New feature value to predict')

    args = parser.parse_args()

    # Load all preprocessing objects
    scaler_x, power_transformer_x, poly, scaler_y = load_preprocessing_objects(
        args.scaler_x_filepath,
        args.power_transformer_x_filepath,
        args.poly_filepath,
        args.scaler_y_filepath
    )

    # Convert new_x_value into numpy array
    new_x = np.array([[args.new_x_value]]).reshape(-1, 1)  # Convert to 2D array for processing

    # Make prediction
    predicted_y = make_inference(args.model_filepath, scaler_x, power_transformer_x, poly, scaler_y, new_x)
    print("Predicted y:", predicted_y.flatten()[0])

