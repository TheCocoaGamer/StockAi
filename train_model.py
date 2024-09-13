import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, PowerTransformer
import matplotlib.pyplot as plt
import pickle
import argparse
import pandas as pd
from tensorflow.keras.losses import MeanSquaredError
def process_and_train(data_path, model_filepath, scaler_x_filepath, power_transformer_x_filepath, poly_filepath, scaler_y_filepath):
    # Load data
    data = pd.read_csv(data_path)
    x = np.array(data['x']).reshape(-1, 1)
    y = np.array(data['y'])

    # Initial normalization
    scaler_x = MinMaxScaler()
    x_normalized = scaler_x.fit_transform(x)
    print("Data normalized.")

    # Applying Power Transform to achieve more Gaussian-like distribution
    power_transformer_x = PowerTransformer()
    x_power_transformed = power_transformer_x.fit_transform(x_normalized)
    print("Power transformation applied.")

    # Applying Polynomial Features
    poly = PolynomialFeatures(degree=16)
    x_poly = poly.fit_transform(x_power_transformed)
    print("Polynomial features generated.")

    # Reshape for LSTM
    x_lstm = x_poly.reshape((x_poly.shape[0], x_poly.shape[1], 1))

    # Scaling y-values
    scaler_y = MinMaxScaler()
    y_transformed = scaler_y.fit_transform(y.reshape(-1, 1))
    print("Target variable scaled.")

    # Save the scalers and polynomial features using pickle
    with open(scaler_x_filepath, 'wb') as file:
        pickle.dump(scaler_x, file)
    with open(power_transformer_x_filepath, 'wb') as file:
        pickle.dump(power_transformer_x, file)
    with open(poly_filepath, 'wb') as file:
        pickle.dump(poly, file)
    with open(scaler_y_filepath, 'wb') as file:
        pickle.dump(scaler_y, file)
    print("Preprocessing objects saved.")

    # Model building
    model = models.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(x_lstm.shape[1], 1)),  # First LSTM layer
        layers.LSTM(20, return_sequences=False),  # Second LSTM layer
        layers.Dense(10, activation='relu'),
        layers.Dense(1)  # Output layer
    ])

    # Optimizer with learning rate
    optimizer = optimizers.Adam(learning_rate=0.001)

    # Model compilation
    model.compile(optimizer=optimizer, loss=MeanSquaredError())

    # Training the model
    print('Training...')
    model.fit(
        x_lstm, 
        y_transformed, 
        epochs=2500, 
        batch_size=128
    )
    print("Model trained.")

    # Save the Keras model
    model.save(model_filepath)
    print(f"Model saved to {model_filepath}.")

    # Predictions
    y_pred_transformed = model.predict(x_lstm)
    y_pred = scaler_y.inverse_transform(y_pred_transformed)  # Transform predictions back to original scale

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color='blue', label='Actual')
    plt.scatter(x, y_pred.flatten(), color='red', label='Predicted')
    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTM model on provided data")
    parser.add_argument('data_path', type=str, help='Path to the CSV data file')
    parser.add_argument('--model_filepath', type=str, default='lstm_model.h5', help='Path to save the trained LSTM model')
    parser.add_argument('--scaler_x_filepath', type=str, default='scaler_x.pkl', help='Path to save the X scaler object')
    parser.add_argument('--power_transformer_x_filepath', type=str, default='power_transformer_x.pkl', help='Path to save the power transformer object')
    parser.add_argument('--poly_filepath', type=str, default='poly.pkl', help='Path to save the polynomial features object')
    parser.add_argument('--scaler_y_filepath', type=str, default='scaler_y.pkl', help='Path to save the Y scaler object')
    args = parser.parse_args()

    process_and_train(args.data_path, args.model_filepath, args.scaler_x_filepath, args.power_transformer_x_filepath, args.poly_filepath, args.scaler_y_filepath)

