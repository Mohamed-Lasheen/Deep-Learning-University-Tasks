import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class preprocessing:
    def __init__(self, feature1, feature2, Class1, Class2):
      self.feature1 = feature1
      self.feature2 = feature2
      self.Class1 = Class1
      self.Class2 = Class2
      
      data = pd.read_csv('Dry_Beans_Dataset.csv')
      data['Perimeter'] = pd.to_numeric(data['Perimeter'].str.replace('٫', '.'), errors='coerce')
      data['MajorAxisLength'] = pd.to_numeric(data['MajorAxisLength'].str.replace('٫', '.'), errors='coerce')
      data['MinorAxisLength'] = pd.to_numeric(data['MinorAxisLength'].str.replace('٫', '.'), errors='coerce')
      data['roundnes'] = pd.to_numeric(data['roundnes'].str.replace('٫', '.'), errors='coerce')
      data["MinorAxisLength"] = data["MinorAxisLength"].fillna(data["MinorAxisLength"].median()) 
      
      data[self.feature1] = np.log2(data[self.feature1]+1)
      data[self.feature2] = np.log2(data[self.feature2]+1)
      
      newdata = pd.DataFrame(data[self.feature1])
      newdata[self.feature2] = data[self.feature2]
      
      columns_to_standardize = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
      scaler = StandardScaler()
      data[columns_to_standardize] = scaler.fit_transform(data[columns_to_standardize])
      
      newdata['Class'] = data['Class']   
        
      self.data1 = newdata[newdata['Class'] == Class1]
      self.data2 = newdata[newdata['Class'] == Class2]
      
      self.data1 = self.data1.replace(Class1,-1)
      self.data2 = self.data2.replace(Class2,1)    

      X_train1, X_test1, y_train1, y_test1 = train_test_split(self.data1.iloc[:, 0:2], self.data1['Class'], test_size=0.4,random_state=1,shuffle=True)
      X_train2, X_test2, y_train2, y_test2 = train_test_split(self.data2.iloc[:, 0:2], self.data2['Class'], test_size=0.4,random_state=1,shuffle=True)

      self.X_train = pd.concat([X_train1, X_train2])
      #######################################
      self.X_test = pd.concat([X_test1, X_test2])
      #######################################
      self.y_train = pd.concat([y_train1, y_train2])
      #######################################
      self.y_test = pd.concat([y_test1, y_test2])

    def splitdata(self):
      return self.X_train, self.X_test, self.y_train, self.y_test