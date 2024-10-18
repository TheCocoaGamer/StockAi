# Imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from train_model import process_and_train
from predict_model import make_inference
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime, timedelta
import csv
import config  # Import the config file

# Variables
AUTHOR = "Malcolm Allen"
VERSION = "2.3.1"
EPOCHS = 20#10
count = 15000#15000
#Filler Variables
MODEL_FILEPATH_ASK = r"Python\My Programs\Bot1\lstm_modelask.h5"
MODEL_FILEPATH_BID = r"Python\My Programs\Bot1\lstm_modelbid.h5"
SCALER_X_FILEPATH = r"Python\My Programs\Bot1\scaler_x.pkl"
POWER_TRANSFORMER_X_FILEPATH = r"Python\My Programs\Bot1\power_transformer_x.pkl"
POLY_FILEPATH = r"Python\My Programs\Bot1\poly.pkl"
SCALER_Y_FILEPATH = r"Python\My Programs\Bot1\scaler_y.pkl"
INSTRUMENT = 'EUR_USD'
BID_PATH = r"Python\My Programs\Bot1\bid_prices.csv"
ASK_PATH = r"Python\My Programs\Bot1\ask_prices.csv"

# Functions

# Create an API client
def setup_client():
    print("Client Setup!")  # Debugging line
    return oandapyV20.API(access_token=config.ACCESS_TOKEN, environment="practice")

# Trains the bot
def train_data(MODEL_FILEPATH, data_path):
    print("Training Data!")  # Debugging line
    process_and_train(EPOCHS, data_path, MODEL_FILEPATH, SCALER_X_FILEPATH, POWER_TRANSFORMER_X_FILEPATH, POLY_FILEPATH, SCALER_Y_FILEPATH)

# Function to get the current time formatted as hhmmss.ms06
def get_time(minutes_from_now=0):
    print("Getting Time!")  # Debugging line
    now = datetime.now() + timedelta(minutes=minutes_from_now)
    fnow = f"{now.hour:02}{now.minute:02}{now.second:02}.{now.microsecond:06}"
    return np.array(fnow).reshape(-1, 1)

# Function to format time
def format_time(time_str):
    print("Formatting Time!")  # Debugging line
    dt = datetime.strptime(time_str[:26], "%Y-%m-%dT%H:%M:%S.%f")
    return dt.strftime("%H%M%S.%f")

# Predicts the data based on a time input with the following: hhmmss.ms06
def pred_data(model_path, time):
    print("Predicting Data!")  # Debugging line
    new_x = np.array([float(time)])  # Ensure new_x is an array
    predicted_y = make_inference(model_path, SCALER_X_FILEPATH, POWER_TRANSFORMER_X_FILEPATH, POLY_FILEPATH, SCALER_Y_FILEPATH, new_x)
    return predicted_y

# Function to write the bid and ask prices to CSV files
def write_csv_files():
    client = setup_client()
    bid_prices, bid_times, ask_prices, ask_times = fetch_prices(count,client)
    # Write bid prices and times to bid_prices_times.csv
    with open(BID_PATH, mode='w', newline='') as bid_file:
        bid_writer = csv.writer(bid_file)
        bid_writer.writerow(['x', 'y'])  # Write header
        for time, price in zip(bid_times, bid_prices):
            bid_writer.writerow([time, price])
    
    # Write ask prices and times to ask_prices_times.csv
    with open(ASK_PATH, mode='w', newline='') as ask_file:
        ask_writer = csv.writer(ask_file)
        ask_writer.writerow(['x', 'y'])  # Write header
        for time, price in zip(ask_times, ask_prices):
            ask_writer.writerow([time, price])

# Function to fetch candles
def fetch_candles(client, instrument, count, granularity, from_time=None, to_time=None):
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Invalid value specified for 'count'. It must be a positive integer.")
    
    params = {
        "count": min(count, 5000),  # Fetch in batches of 5000
        "granularity": granularity,
        "price": "BA"  # Request bid and ask prices
    }
    if from_time:
        params["from"] = from_time
    if to_time:
        params["to"] = to_time

    request = instruments.InstrumentsCandles(instrument=instrument, params=params)
    response = client.request(request)
    print("Candles fetched!")  # Debugging line
    return response

# Function to fetch prices
def fetch_prices(count, client):
    granularity = "M1"  # Example granularity
    bid_prices = []
    ask_prices = []
    bid_times = []
    ask_times = []
    
    fetched_count = 0
    while fetched_count < count:
        count = min(count - fetched_count, 5000)
        candles = fetch_candles(client, INSTRUMENT, count, granularity)
        
        for candle in candles['candles']:
            if 'bid' in candle and 'ask' in candle:
                bid_prices.append(candle['bid']['c'])
                ask_prices.append(candle['ask']['c'])
                formatted_time = datetime.strptime(candle['time'], "%Y-%m-%dT%H:%M:%S.%f000Z").strftime("%Y%m%d%H%M%S.%f")
                bid_times.append(formatted_time)
                ask_times.append(formatted_time)
        
        fetched_count += count
    
    print("Last 5 Bid prices:", bid_prices[-5:])  # Print last 5 values
    print("Last 5 Ask prices:", ask_prices[-5:])  # Print last 5 values
    print("Last 5 Bid times:", bid_times[-5:])    # Print last 5 values
    print("Last 5 Ask times:", ask_times[-5:])    # Print last 5 values
    print(len(ask_prices))
    print(len(bid_prices))



    return bid_prices, bid_times, ask_prices, ask_times

# Function to start the bot with Bid model
def startupBid():
    write_csv_files()
    train_data(MODEL_FILEPATH_BID, BID_PATH)

# Function to start the bot with Ask model
def startupAsk():
    write_csv_files()
    train_data(MODEL_FILEPATH_ASK, ASK_PATH)

def make_prediction(MODEL_FILEPATH, time=0):
    print(time, " Minutes ASK prediction: ", get_time(time), " Prediction: ", pred_data(MODEL_FILEPATH, get_time(time)))
    return pred_data(MODEL_FILEPATH, get_time(time))

def main():
    startupAsk()
    startupBid()
    make_prediction(MODEL_FILEPATH_ASK, 30)

    
if __name__ == "__main__":
    main()

# End of file