from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

LOOKBACK_DAYS = 60
TRAINING_EPOCHS = 2
FORECAST_DAYS = 30
HISTORY_START_DATE = "2000-01-01"


def download_prices(ticker: str) -> pd.Series:
    """Download adjusted daily closing prices for a ticker."""
    temp_root = Path(tempfile.gettempdir())
    if not str(temp_root).isascii():
        raise RuntimeError(
            "The temporary folder path must contain only ASCII characters "
            "to load the CA certificate on Windows."
        )

    with tempfile.TemporaryDirectory(dir=temp_root) as cert_dir:
        cert_path = Path(cert_dir) / "cacert.pem"
        shutil.copyfile(certifi.where(), cert_path)
        session = requests.Session(impersonate="chrome", verify=str(cert_path))
        data = yf.download(
            ticker,
            start=HISTORY_START_DATE,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            session=session,
        )

    if data.empty or "Close" not in data:
        raise ValueError(
            f"No price data was returned for {ticker}. Check the ticker "
            "and your network connection."
        )

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype("float64")

    if len(close) <= LOOKBACK_DAYS:
        raise ValueError(
            f"Only {len(close)} daily prices were returned for {ticker}; "
            f"at least {LOOKBACK_DAYS + 1} are needed to train the model."
        )

    close.name = "Close"
    return close


def train_and_forecast(
    prices: pd.Series,
    forecast_days: int = FORECAST_DAYS,
) -> tuple[pd.Series, float]:
    """Train the notebook's LSTM model and forecast future trading weekdays."""
    if forecast_days < 1:
        raise ValueError("The forecast period must be at least one day.")
    if len(prices) <= LOOKBACK_DAYS:
        raise ValueError(
            f"At least {LOOKBACK_DAYS + 1} daily prices are needed to train."
        )

    values = prices.to_numpy(dtype=np.float32).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    x = np.stack(
        [scaled[i : i + LOOKBACK_DAYS, 0] for i in range(len(scaled) - LOOKBACK_DAYS)]
    ).reshape(-1, LOOKBACK_DAYS, 1)
    y = scaled[LOOKBACK_DAYS:, 0]

    model = Sequential(
        [
            Input(shape=(LOOKBACK_DAYS, 1)),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dropout(0.3),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(x, y, epochs=TRAINING_EPOCHS, batch_size=8, verbose=0)

    latest_window = scaled[-LOOKBACK_DAYS:].copy()
    predictions: list[float] = []
    for _ in range(forecast_days):
        prediction = model.predict(
            latest_window.reshape(1, LOOKBACK_DAYS, 1),
            verbose=0,
        )
        predictions.append(float(scaler.inverse_transform(prediction)[0, 0]))
        latest_window = np.concatenate((latest_window[1:], prediction), axis=0)

    first_forecast_day = prices.index[-1] + pd.Timedelta(days=1)
    forecast_dates = pd.bdate_range(start=first_forecast_day, periods=forecast_days)
    forecast = pd.Series(predictions, index=forecast_dates, name="Forecast")
    final_loss = float(history.history["loss"][-1])
    return forecast, final_loss
