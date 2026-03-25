import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("🚀 AI Stock Price Prediction")
st.write("Real-time stock prediction app")

# Sidebar
st.sidebar.title("📌 Controls")

stock_options = {
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Nippon India": "NIPPOIND.NS"
}

selected_company = st.sidebar.selectbox("Select Company", list(stock_options.keys()))
custom_stock = st.sidebar.text_input("Or Enter Stock Symbol")

stock_name = custom_stock if custom_stock else stock_options[selected_company]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# 🔥 NO BUTTON — AUTO RUNS
data = yf.download(stock_name, start=start_date, end=end_date)

if data.empty:
    st.error("❌ Invalid stock or no data available")
else:
    data = data[['Close']]

    st.subheader(f"📊 {stock_name} Stock Data")
    st.dataframe(data.tail())

    # Live chart
    st.subheader("📈 Live Stock Trend")
    st.line_chart(data['Close'])

    # ML Prediction
    future_days = 10
    data['Prediction'] = data['Close'].shift(-future_days)

    X = np.array(data[['Close']])[:-future_days]
    y = np.array(data['Prediction'])[:-future_days]

    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.metric("📉 Model RMSE", f"{rmse:.2f}")

        # Future predictions
        x_future = np.array(data[['Close']].tail(future_days))
        future_predictions = model.predict(x_future)

        st.subheader("🔮 Next 10 Days Prediction")
        st.write(future_predictions)

        # Graph 1
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, y_pred)
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.set_title("Actual vs Predicted")
        st.pyplot(fig1)

        # Graph 2
        fig2, ax2 = plt.subplots()
        ax2.plot(data['Close'], label="Actual")

        future_index = np.arange(len(data), len(data) + future_days)
        ax2.plot(future_index, future_predictions, linestyle='dashed', label="Predicted")

        ax2.legend()
        ax2.set_title("Trend + Prediction")
        st.pyplot(fig2)