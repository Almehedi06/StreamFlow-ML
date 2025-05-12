import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_and_scale(df, features, target):
    X = df[features].values
    y = df[target].values.ravel()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def reshape_for_sequence(X, timesteps):
    return np.array([X[i - timesteps:i] for i in range(timesteps, len(X))])


def load_data(train_path, dev_path):
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    return train_df, dev_df
