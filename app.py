import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import yfinance as yf

from main import predict

with open('model.pkl',"rb") as pkl:
    model, scaler = pickle.load(pkl)

def main():
    st.title("Real-Time Stock Trend Predictor")
    st.write("Predicts next-day movement using machine learning")
    ticker = st.text_input("Enter Enter stock ticker (e.g. AAPL, TSLA, RELIANCE.NS)")

    if st.button('Predict'):
        try:
            data = yf.download(ticker, period='10d')
        except:
            st.error("Falied to fetch data.")
            
        if data is not None and not data.empty:    
            df = data[['Close']].dropna()

            if len(df)<5:
                st.error("Not enough prices! Please try another ticker")
            else:
                y_pred, probability= predict(df, model, scaler)
                st.write(f"Ticker: {ticker}")
                st.write(f"Latest Close: {df['Close'].iloc[-1].item():.2f}")

                st.markdown("### 📊 Prediction Result")
                if y_pred[0]==1:
                    st.success("The price is likely to go UP")            
                else:
                    st.error("The price is likely to go DOWN")
                
                st.line_chart(df['Close'])

            
    # st.write("Input last 3 closing prices")
    # a = st.number_input('Closing Price 1:', value=0.0)
    # b = st.number_input('Closing Price 2:', value=0.0)
    # c = st.number_input('Closing Price 3:', value=0.0)
    # d = st.number_input('Closing Price 4:', value=0.0)
    # e = st.number_input('Closing Price 5:', value=0.0)

    # df = pd.DataFrame([a, b, c, d, e ], columns=["Close"])

    # if st.button('Predict'):
    #     y_pred, probability= predict(df, model, scaler)
    #     st.markdown("### 📊 Prediction Result")
    #     if y_pred[0]==1:
    #         st.success("The price is likely to go UP")
    #         # st.write(f"Confidence:{probability[:,1][0]*100:0.3f}%")
    #         # st.progress(probability[:,1][0])
    #     else:
    #         st.error("The price is likely to go DOWN")
    #         # st.write(f"Confidence:{probability[:,0][0]*100:0.3f}%")
    #         # st.progress(probability[:,1][0])

    #     st.line_chart(df['Close'])

    #     df_plot = df.copy()
    #     df_plot['Day'] = df_plot.index.to_series().astype(float) + 1.0

    #     fig = px.line(
    #         df_plot,
    #         x='Day',
    #         y='Close',
    #         markers=True,
    #         title='Closing Prices',
    #         labels={'Day': 'Day', 'Close': 'Closing Price'}
    #     )
    #     fig.update_xaxes(type='linear', tickmode='linear', tick0=1, dtick=1)
    #     st.plotly_chart(fig, width='stretch')


if __name__ == '__main__':
    main()