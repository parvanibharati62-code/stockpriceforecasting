# Stock Price Forecaster

A Streamlit web app that downloads historical stock prices, trains an LSTM
model, and plots a 30-weekday forecast. Enter a ticker such as `AAPL`, submit
the form, and the app displays the recent historical prices alongside the
forecast.

**Live app:** [Open Stock Price Forecaster](https://web-production-1802606e.up.railway.app/)

> **Disclaimer:** This project is for educational purposes only. Forecasts are
> model estimates, not financial advice or a recommendation to buy or sell
> securities.

## Features

- Accepts stock ticker symbols through a simple Streamlit interface.
- Downloads daily adjusted closing prices from Yahoo Finance, starting in 2000
  when data is available.
- Trains a two-layer LSTM model on the ticker's available price history.
- Displays a chart of the latest year of historical prices and a 30-weekday
  forecast.
- Shows the latest close, final forecast estimate, percentage difference, and
  training loss.
- Includes error messages for invalid tickers, failed downloads, and
  insufficient price history.

## How it works

1. The app downloads daily prices for the submitted ticker using `yfinance`.
2. The closing-price series is scaled and split into 60-day sequences.
3. A Keras LSTM model is trained for two epochs.
4. The model iteratively predicts 30 future weekdays.
5. Streamlit displays the forecast with recent historical prices.

Forecast dates skip weekends but **do not** account for exchange holidays.
Each submission trains a new model; the app does not save trained models or
user data. The forecast is a simple iterative model output and should not be
treated as a reliable prediction of future market prices.

## Project structure

```text
.
├── app.py                 # Streamlit interface
├── forecasting.py         # Data download, LSTM training, and prediction
├── requirements.txt       # Python dependencies
├── Procfile               # Railway start command
├── .python-version        # Python version used for deployment
└── notebooks/
    └── StockPrice.ipynb   # Original exploration notebook
```

## Run locally

Python 3.12 is recommended (the deployment configuration uses it).

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) when the app
starts. Open that URL in your browser.

## Deploy to Railway

1. Push the project to a GitHub repository. Include `app.py`,
   `forecasting.py`, `requirements.txt`, `Procfile`, and `.python-version`.
   Do not commit the local `venv` or `.venv` directories.
2. In Railway, create a project and choose **Deploy from GitHub repo**.
3. Select the repository and use the repository root as the service root
   directory.
4. Railway installs the dependencies from `requirements.txt` and starts the
   app using the `Procfile`. The app binds to Railway's assigned `$PORT`.
5. After the service deploys, open **Settings → Networking** and generate a
   public domain.

The initial build can take several minutes because TensorFlow is a large
dependency. Check the Railway build and deployment logs if the service does
not become healthy.

## Dependencies

- Streamlit for the web interface
- TensorFlow/Keras for the LSTM model
- yfinance and curl-cffi for market data retrieval
- pandas, NumPy, and scikit-learn for data preparation
- Matplotlib for the forecast chart

## Data and limitations

- Price history is provided by Yahoo Finance; availability and ticker formats
  depend on that service.
- The app requires an internet connection to retrieve price data.
- A ticker needs at least 61 valid daily closing prices to train the model.
- The displayed percentage compares the latest downloaded close with the final
  forecast estimate; it is not a guaranteed return.
- Training loss is the model's final mean squared error on its training data,
  not a measure of expected future accuracy.
