import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class PreProcessing:
      def __init__(self):
            data = pd.read_csv('Dry_Beans_Dataset.csv')
            data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
            data['Perimeter'] = pd.to_numeric(data['Perimeter'].str.replace('٫', '.'), errors='coerce')
            data['MajorAxisLength'] = pd.to_numeric(data['MajorAxisLength'].str.replace('٫', '.'), errors='coerce')
            data['MinorAxisLength'] = pd.to_numeric(data['MinorAxisLength'].str.replace('٫', '.'), errors='coerce')
            data['roundnes'] = pd.to_numeric(data['roundnes'].str.replace('٫', '.'), errors='coerce')
            data["MinorAxisLength"] = data["MinorAxisLength"].fillna(data["MinorAxisLength"].median())
            X = data.drop('Class', axis=1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y = pd.get_dummies(data['Class'])
            self.X_train1, self.X_test1, self.y_train1, self.y_test1 = train_test_split(X_scaled, y,
                                                                                        test_size=0.4, random_state=42,
                                                                                        shuffle=True, stratify=y)




