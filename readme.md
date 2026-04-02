### Stock Trend Predictor
A machine learning project that predicts whether a stock's price will move up or down the next day. Built using scikit-learn and deployed as an interactive web app with Streamlit.
This project was developed to explore how machine learning performs on real financial time series data. The model achieves ~50–53% accuracy, which reflects the inherent difficulty of predicting markets rather than model failure.

<!-- ## Live Demo
[Open the Streamlit App](YOUR_LINK_HERE) -->

## Overview
Users can input a stock ticker (e.g., AAPL, TSLA), and the app:
- Fetches one year of historical price data
- Applies a feature engineering pipeline based on technical indicators
- Predicts whether the next day’s closing price will be higher or lower
- Displays current signals and model output
- Compares a strategy based on model predictions against a buy-and-hold baseline

### Feature Engineering
Rather than feeding raw prices into the model, we compute a set of technical indicators that traders actually use. Here's what each one means:
**Lagged Returns** : The percentage change in price from one day to the next. We feed in the last 3 days of returns so the model can see recent momentum. If the stock went up 1% yesterday and 2% the day before, those are signals about the current trend.
**Rolling Volatility (3-day and 5-day)** How much the stock's returns have been jumping around recently. A stock that moves 3% up, 3% down, 3% up is more volatile than one that moves 0.1% consistently. We measure this as the standard deviation of returns over a short window. High volatility can signal uncertainty or big news events.
**SMA Crossover (Simple Moving Average)** A moving average smooths out daily noise by averaging the last N days of prices. We compute two: a 5-day (short term) and a 20-day (long term). When the short term average crosses above the long term average, it suggests the recent trend is gaining strength. This is one of the oldest signals in technical analysis. We turn it into a simple 1 (bullish) or 0 (bearish) flag.
**RSI (Relative Strength Index)** RSI answers: over the last 14 days, how strong were the up days compared to the down days? It produces a number between 0 and 100. Above 70 usually means the stock has been going up a lot and might be due for a pullback (overbought). Below 30 means it has been falling and might bounce back (oversold). Between 40 and 60 means neutral momentum, which is what most stocks look like most of the time.
**Price Position in 10-day Range** Where is today's closing price relative to the highest and lowest prices of the last 10 days? A value near 1 means the stock is trading near its recent high. A value near 0 means it is near its recent low. This gives the model context about where the price sits in its recent range without depending on the raw price value itself.
**Price Deviation from 3-day Mean** How far today's price is from the average of the last 3 days. This captures very short term displacement, whether the stock has made a sudden move away from where it has been sitting.

### Model
- Logistic Regression with balanced class weights  
- Train/test split: 80/20 (chronological, no shuffling)  
- Designed to respect the time-series nature of financial data  

## Evaluation
Model performance is evaluated using:
- Accuracy
- Precision / Recall / F1-score
- ROC-AUC (~0.55)

Accuracy remains close to random (~50–53%), which is consistent with the **Efficient Market Hypothesis**—suggesting that price movements are difficult to predict using historical price data alone.

## Backtesting
The app simulates a simple trading strategy:

- Predict **Up** → take the next day’s return  
- Predict **Down** → remain in cash  

Performance is compared against a buy-and-hold strategy using cumulative returns starting from an initial value of $1.

### Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Stack
Python, scikit-learn, pandas, numpy, yfinance, Streamlit, Plotly




