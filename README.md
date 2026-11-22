# Stock Price Forecaster

A Streamlit app based on `notebooks/StockPrice.ipynb`. Enter a ticker such as
`AAPL` to download adjusted daily prices, train the notebook's two-layer LSTM,
and view a 30-trading-day forecast.

## Run locally

Use Python 3.12, then install and launch:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Railway

1. Push this project to a GitHub repository.
2. In Railway, create a new project and deploy from that repository.
3. Set the service root directory to the project root (if prompted).
4. Railway uses `requirements.txt`, `.python-version`, and `Procfile` to install
   dependencies and start Streamlit on Railway's assigned port.
5. Once deployment is healthy, generate a public domain from the service's
   **Settings → Networking** section.

The first build downloads TensorFlow and may take several minutes. The app
trains a fresh model when a ticker is submitted. Forecasts are illustrative
and are not financial advice.
