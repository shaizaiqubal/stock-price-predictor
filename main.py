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
    df_copy = df_copy.dropna().copy()
    df_copy = df_copy.reset_index(drop=True)

    return_l = df_copy['returns'].tolist()
    close_l = df_copy['Close'].tolist()

    X = []
    y = []

    for i in range(2, len(return_l) - 1):
        f = [
            return_l[i-2], return_l[i-1], return_l[i],
            np.std([return_l[i-2], return_l[i-1], return_l[i]],ddof=1),
            close_l[i]-np.mean([close_l[i-2],close_l[i-1],close_l[i]])
        ]
        X.append(f)

        y.append(int(close_l[i+1] >= close_l[i]))


    X = np.array(X)
    y = np.array(y)

    return X,y

def train_model(df):
    X,y= build_features(df)
    X_train, _, y_train, _ = train_test_split(X,y, test_size=0.2, shuffle=False)
    scaler=StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model=LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    with open('model.pkl','wb') as pkl:
        pickle.dump((model,scaler),pkl)
    return model, scaler

def predict(df,model,scaler):

    X,_=build_features(df)
    X_test=X[-1].reshape(1,-1)
    X_test = scaler.transform(X_test)
    y_pred=model.predict(X_test)
    probability=model.predict_proba(X_test)

    return y_pred,probability
