import numpy as np
import pandas as pd

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_features(df):

    df_copy = df[['Close']].copy()
    df_copy.columns = ['Close']
    df_copy['prev_close']=df_copy['Close'].shift(1)

    df_copy['returns']=(df_copy['Close']-df_copy['prev_close'])/df_copy['prev_close']

    df_copy['volatility_5'] = df_copy['returns'].rolling(5).std()
    df_copy['volatility_3'] = df_copy['returns'].rolling(3).std()

    df_copy['sma_5'] = df_copy['Close'].rolling(5).mean()
    df_copy['sma_20'] = df_copy['Close'].rolling(20).mean()
    df_copy['sma_crossover'] = (df_copy['sma_5'] > df_copy['sma_20']).astype(int)

    delta = df_copy['Close'].diff()          # daily change in price
    gain = delta.clip(lower=0)          # keep only positive changes
    loss = -delta.clip(upper=0)         # keep only negative changes (flip sign)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / (avg_loss + 1e-9)   # ratio of avg gain to avg loss
    df_copy['rsi'] = 100 - (100 / (1 + rs))

    high_10 = df_copy['Close'].rolling(10).max()
    low_10 = df_copy['Close'].rolling(10).min()

    df_copy['price_position'] = (df_copy['Close'] - low_10) / (high_10 - low_10 + 1e-9)

    df_copy = df_copy.dropna().copy()
    df_copy = df_copy.reset_index(drop=True)

    return_l = df_copy['returns'].tolist()
    close_l = df_copy['Close'].tolist()
    vol_3l = df_copy['volatility_3'].tolist()
    vol_5l = df_copy['volatility_5'].tolist()
    rsi_l = df_copy['rsi'].tolist()
    sma_l = df_copy['sma_crossover'].tolist()
    pp_l = df_copy['price_position'].tolist()


    X = []
    y = []

    for i in range(2, len(return_l) - 1):
        f = [
            return_l[i-2], return_l[i-1], return_l[i],
            close_l[i]-np.mean([close_l[i-2],close_l[i-1],close_l[i]]),
            vol_3l[i], vol_5l[i], rsi_l[i], sma_l[i], pp_l[i]
        ]
        X.append(f)

        y.append(int(close_l[i+1] >= close_l[i]))


    X = np.array(X)
    y = np.array(y)

    return X,y, df_copy

def train_model(df):
    X,y,_= build_features(df)
    X_train, _, y_train, _ = train_test_split(X,y, test_size=0.2, shuffle=False)
    scaler=StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model=LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    with open('model.pkl','wb') as pkl:
        pickle.dump((model,scaler),pkl)
    return model, scaler

def predict(df,model,scaler):

    X,_,features_df=build_features(df)
    X_test=X[-1].reshape(1,-1)
    X_test = scaler.transform(X_test)
    y_pred=model.predict(X_test)
    probability=model.predict_proba(X_test)

    return y_pred, probability, features_df
