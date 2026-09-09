import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
iris=load_iris()
x=iris.data
y=iris.target
df=pd.DataFrame(x,y)
print(df.isnull().sum())
scaler=StandardScaler()
minmax=MinMaxScaler()
x_scaled=scaler.fit_transform(x)
x_minmax=minmax.fit_transform(x)

print(x_scaled)
print(x_minmax)
