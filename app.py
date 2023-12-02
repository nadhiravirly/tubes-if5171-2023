from flask import Flask, render_template, request
from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models.models import Base, Price  # Import your SQLAlchemy model
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

scaler_X = MinMaxScaler(feature_range=(0, 1))

# Configure the database
engine = create_engine('sqlite:///price.db')
Base.metadata.bind = engine

@app.route('/')
def index():
    return render_template('index.html')

def get_data():
    try :
        db_path = 'price.db'
        
        # Buat koneksi ke database menggunakan SQLAlchemy
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Baca seluruh data dari tabel ke dalam DataFrame
        query = 'SELECT * FROM price' 
        df = pd.read_sql(query, engine)
        df = pd.DataFrame(df)
        return df
    except Exception as e:
        print(f"Error access db: {e}")

def normalized(sequence_data):
    middle_matrix = sequence_data[:, 1]
    print(middle_matrix)
    middle_matrix = middle_matrix.astype(float)
    print(middle_matrix)
    middle_matrix = middle_matrix.reshape(1, -1)
    print(middle_matrix)
    X_normalized = scaler_X.fit_transform(middle_matrix.reshape(-1, 1)).reshape(middle_matrix.shape + (1,))
    print(X_normalized)
    X_normalized = X_normalized.astype(np.float32)
    print(X_normalized)

    return X_normalized

def preprocess(days_difference):

    try:
        print("Fetching data...")
        df = get_data()
        print(df.head())
        print(type(df))
        print("Data fetched successfully.")

        print("Converting 'date' column to datetime...")
        df['date'] = pd.to_datetime(df['date'])
        print("'date' column converted successfully.")

        print("Converting numeric columns...")
        numeric_columns = ['close']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        print("Numeric columns converted successfully.")

        print("Checking for rows with null values...")
        df = df.fillna(df.mean())
        print("Rows with null values updated with mean.")

        print("Selecting columns...")
        df = df[['date', 'close']]
        print("Columns selected successfully.")

        print("Create sequence started...")
        df_sorted = df.sort_values(by='date', ascending=False)
        window = 3
        sequence_data = df_sorted.values[:window]
        print("Create sequence successfully.")

        print("Select last date started...")
        last_date = df['date'][:1]
        last_date = pd.Timestamp(last_date.iloc[0])
        print("Select last date successfully.")

        stat = 2

        return sequence_data, last_date, stat

    except Exception as e:
        print(f"Error in pre-process: {e}")
        return -2  # or return an error code or message

def next_weekday(d):
    while d.weekday() in {5, 6}:  # 5 adalah Sabtu, 6 adalah Minggu
        d += pd.DateOffset(days=1)
    return d


def denormalized(forecasted_values):
    reverse_forecast =  scaler_X.inverse_transform(forecasted_values.reshape(-1, 1)).flatten()
    return reverse_forecast


# Function to load LSTM model and make predictions
def lstm_predict(num_predict):

    num_days_to_forecast = num_predict

    stat = 0 

    # Initialize an array to store the forecasted values
    forecasted_values = []

    try:
        loaded_model = load_model('models/model_lstm_s2.h5')
        stat = 1  # Update stat if the model is loaded successfully
        print("preprocess started...")
        sequence_data, last_date, stat = preprocess(num_days_to_forecast)
        print("preprocess ended.")
        print("normalization started...")
        X_normalized = normalized(sequence_data)
        print("normalization ended.")


        # Use the last few days from the test data to start the forecasting
        input_sequence = X_normalized

        for _ in range(num_days_to_forecast):
            # Make a prediction for the next day
            next_day_prediction = loaded_model.predict(input_sequence).flatten()[0]

            # Store the prediction in the forecasted_values array
            forecasted_values.append(next_day_prediction)

            # Update the input_sequence for the next prediction
            next_day_prediction = np.array([[next_day_prediction]])
            input_sequence = np.concatenate([input_sequence[:, 1:], np.expand_dims(next_day_prediction, axis=1)], axis=1)

        # Convert forecasted_values to a numpy array
        forecasted_values = np.array(forecasted_values)
        print(forecasted_values)

        # Create dates for the forecasted values
        forecast_dates = pd.date_range(start=last_date, periods=num_days_to_forecast + 1, freq='B').map(next_weekday)[1:]

        # Create a DataFrame to store the forecasted values and dates
        forecast_df = pd.DataFrame({
         'Date': forecast_dates,
            'Forecast': forecasted_values
        })

        # Display the forecast DataFrame
        print("forecast : ", forecast_df)
        print(forecast_df)

        reverse_forecast = denormalized(forecasted_values)

        reverse_forecast_df = pd.DataFrame({
            'Date': forecast_df['Date'],
            'Reverse_Forecast': reverse_forecast
        })

        print("reverse_forecast_df: ",reverse_forecast_df)
        

        stat = 3

        print("end of process LSTM")
        return reverse_forecast_df, stat

    except Exception as e:
        print(f"Error predicting using LSTM: {e}")
        stat = -2
        return forecast_df, stat


@app.route('/predict', methods=['POST'])
def search():
    try:
        # Get the inputted numeric from the form
        num_predict = int(request.form['num_predict']) 
        print("num_predict = ", num_predict)

        lstm_result, stat = lstm_predict(num_predict)
        result = lstm_result.to_html(index=False)

        return render_template('index.html', num_predict=num_predict, prediction=result, status=stat)

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)


