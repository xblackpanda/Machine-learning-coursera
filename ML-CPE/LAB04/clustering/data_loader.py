import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):

    df = pd.read_csv(file_path)

    
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df