import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

from model import predict

with open('model.pkl',"rb") as pkl:
    model, scaler = pickle.load(pkl)

def main():
    st.title("Stock Price trend Predictor")
    st.write("Input last 3 closing prices")
    a = st.number_input('Closing Price 1:', value=0.0)
    b = st.number_input('Closing Price 2:', value=0.0)
    c = st.number_input('Closing Price 3:', value=0.0)
    d = st.number_input('Closing Price 4:', value=0.0)
    e = st.number_input('Closing Price 5:', value=0.0)

    df = pd.DataFrame([a, b, c, d, e ], columns=["Close"])

    if st.button('Predict'):
        y_pred, probability= predict(df, model, scaler)
        st.markdown("### 📊 Prediction Result")
        if y_pred[0]==1:
            st.success("The price is likely to go UP")
            # st.write(f"Confidence:{probability[:,1][0]*100:0.3f}%")
            # st.progress(probability[:,1][0])
        else:
            st.error("The price is likely to go DOWN")
            # st.write(f"Confidence:{probability[:,0][0]*100:0.3f}%")
            # st.progress(probability[:,1][0])

        
        df_plot = df.copy()
        df_plot['Day'] = df_plot.index.to_series().astype(float) + 1.0

        fig = px.line(
            df_plot,
            x='Day',
            y='Close',
            markers=True,
            title='Closing Prices',
            labels={'Day': 'Day', 'Close': 'Closing Price'}
        )
        fig.update_xaxes(type='linear', tickmode='linear', tick0=1, dtick=1)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()