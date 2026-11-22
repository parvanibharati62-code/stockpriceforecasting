from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from forecasting import download_prices, train_and_forecast

st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📈",
    layout="wide",
)

st.title("Stock Price Forecaster")
st.write(
    "Enter a stock ticker to train an LSTM model on its historical prices "
    "and generate a 30-trading-day forecast."
)

with st.form("forecast_form"):
    ticker_input = st.text_input("Stock ticker", value="AAPL", max_chars=20)
    submitted = st.form_submit_button("Train model and forecast", type="primary")

if submitted:
    ticker = ticker_input.strip().upper()
    if not ticker:
        st.error("Enter a stock ticker to continue.")
        st.session_state.pop("forecast_result", None)
    else:
        st.session_state.pop("forecast_result", None)
        try:
            with st.spinner(f"Downloading prices and training the model for {ticker}..."):
                prices = download_prices(ticker)
                forecast, loss = train_and_forecast(prices)
            st.session_state["forecast_result"] = {
                "ticker": ticker,
                "prices": prices,
                "forecast": forecast,
                "loss": loss,
            }
        except (ValueError, RuntimeError) as error:
            st.error(str(error))

result = st.session_state.get("forecast_result")
if result:
    ticker = result["ticker"]
    prices: pd.Series = result["prices"]
    forecast: pd.Series = result["forecast"]

    last_price = float(prices.iloc[-1])
    forecast_price = float(forecast.iloc[-1])
    change_percent = (forecast_price / last_price - 1) * 100

    st.subheader(f"{ticker} forecast")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Latest close", f"${last_price:,.2f}")
    metric_columns[1].metric(
        f"Forecast in {len(forecast)} trading days",
        f"${forecast_price:,.2f}",
        f"{change_percent:+.2f}%",
    )
    metric_columns[2].metric("Training loss (MSE)", f"{result['loss']:.6f}")

    chart_prices = prices.tail(252)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(chart_prices.index, chart_prices, label="Historical close", color="#2563eb")
    axis.plot(
        forecast.index,
        forecast,
        label="LSTM forecast",
        color="#f97316",
        linestyle="--",
    )
    axis.axvline(forecast.index[0], color="#64748b", linestyle=":", linewidth=1)
    axis.set_title(f"{ticker}: recent prices and 30-trading-day forecast")
    axis.set_xlabel("Date")
    axis.set_ylabel("Adjusted closing price (USD)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.caption(
        "Forecast dates are weekdays and do not account for exchange holidays. "
        "This model is an educational example, not financial advice."
    )
