import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

filename = "./data/09_irisdata.csv"
column_names = ['sepal-lenght', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv(filename, names=column_names)

print("데이터 셋의 행렬 크기:", data.shape)

print("데이터 셋의 요약")
print(data.describe())

print("데이터 셋의 클래스 종류")
print(data.groupby('class').size())

scatter_matrix(data)
plt.savefig('scatter.matrix.png')
plt.show()

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

model = DecisionTreeClassifier()

kfold = KFold(n_splits=10, random_state=5, shuffle=True)
results = cross_val_score(model, x, y, cv=kfold)

print(results.mean())
