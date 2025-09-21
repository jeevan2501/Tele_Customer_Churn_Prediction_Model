import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataPreprocessing:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_clean_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(['customerID'], axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()
        for column in df.select_dtypes(include=['object']).columns:
            if column != 'Churn':
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
        le = LabelEncoder()
        df['Churn'] = le.fit_transform(df['Churn'])
        X = df.drop(['Churn'], axis=1)
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test
