import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import yfinance as yf
from sklearn.model_selection import train_test_split

from main import predict, build_features

with open('model.pkl',"rb") as pkl:
    model, scaler = pickle.load(pkl)

def run_backtest(df):
    X, y, features_df = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_test_scaled = scaler.transform(X_test)
    close_l = df['Close'].values.flatten()
    train_size = len(X_train)
    strategy_returns, bah_returns = [], []
    for i in range(len(X_test)):
        actual_ret = float((close_l[train_size + i + 1] - close_l[train_size + i]) / close_l[train_size + i])
        ypr = model.predict([X_test_scaled[i]])[0]
        strategy_returns.append(actual_ret if ypr == 1 else 0)
        bah_returns.append(actual_ret)

    return [1] + list(np.cumprod([1 + r for r in strategy_returns])), \
    [1] + list(np.cumprod([1 + r for r in bah_returns]))


def main():
    st.title("Real-Time Stock Trend Predictor")
    st.write("Predicts next-day movement using machine learning")
    ticker_options = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "NFLX", "JPM", "Custom"]
    ticker = st.selectbox("Select a stock ticker", options=ticker_options)
    if ticker == "Custom":
        ticker = st.text_input("Enter custom stock ticker")

    if st.button('Predict'):
        try:
            data = yf.download(ticker, period='1y')
        except:
            st.error("Falied to fetch data.")
            
        if data is not None and not data.empty:    
            df = data[['Close']].dropna()

            if len(df)<5:
                st.error("Not enough prices! Please try another ticker")
            else:
                y_pred, probability, features_df= predict(df, model, scaler)

                #Stats
                st.markdown(f"### 📈 {ticker}: Current Market Signals")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Latest Close", f"${df['Close'].iloc[-1].item():.2f}")
                col2.metric("RSI", f"{features_df['rsi'].iloc[-1]:.1f}")
                col3.metric("Price Position", f"{features_df['price_position'].iloc[-1]:.2f}")
                col4.metric("SMA Crossover", "Bullish" if features_df['sma_crossover'].iloc[-1] == 1 else "Bearish")

                #prediction
                st.markdown("### 📊 Prediction Result")
                if y_pred[0]==1:
                    st.success("The price is likely to go UP")            
                else:
                    st.error("The price is likely to go DOWN")
                
                prob = probability[0][1] if y_pred[0] == 1 else probability[0][0]
                #st.write(f"Confidence: {prob*100:.1f}%")

                st.markdown("### 💹 Recent Price")
                st.line_chart(df['Close'])

                #Backtest
                st.markdown("### 🔁 Backtest: Model vs Buy & Hold")
                strategy, bah = run_backtest(df)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=strategy, name='Model Strategy'))
                fig.add_trace(go.Scatter(y=bah, name='Buy & Hold'))
                fig.update_layout(xaxis_title='Trading Days', yaxis_title='Portfolio Value ($1 start)')
                st.plotly_chart(fig)


            
if __name__ == '__main__':
    main()