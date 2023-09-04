#!/usr/bin/env python
# coding: utf-8

# In[3]:


#--------變更工作路徑-------------
import os
path = os.path.abspath(os.path.dirname(__file__))
print(path)
os.chdir(path)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import Preprocess

# 資料前處理
Preprocess.Preprocessing()
print()

# 判讀是否有使用特徵選取
if Preprocess.checkUsingFS():
    data = pd.read_csv('dataset.csv')
else:
    data1 = pd.read_csv('train1.csv')
    data2 = pd.read_csv('test1.csv')
    data = pd.concat([data1,data2])

# 設定 feature和 Target
X = data.drop("price",axis = 1)
y = data["price"]

# 分割資料集為訓練集和測試集
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)

# model 
lr = LinearRegression()
dtree = DecisionTreeRegressor()
regr = RandomForestRegressor()
xgbr = XGBRegressor()

# Fitting X_train and y_train
lr.fit(X_train, y_train)
dtree.fit(X_train, y_train)
regr.fit(X_train, y_train)
xgbr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
print("線性回歸：r2_score: ", r2_score(y_test, lr_pred) * 100)

dtree_pred = dtree.predict(X_test)
print("決策樹  ：r2_score: ", r2_score(y_test, dtree_pred) * 100)

regr_pred = regr.predict(X_test)
print("隨機森林：r2_score: ", r2_score(y_test, regr_pred) * 100)

xgbr_pred = xgbr.predict(X_test)
print("XGBoost：r2_score: ", r2_score(y_test, xgbr_pred) * 100)
#print("預測值", xgbr_pred)

# 隨機森林可視覺化圖形
fig = plt.figure(num=1)
plt.subplot(211)
plt.scatter(range(len(y_test)), y_test, color='b', label='Actual')
plt.scatter(range(len(regr_pred)), regr_pred, color='r', label='Predicted')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.title('Comparison of Actual and Predicted Values')
plt.legend()
# plt.show()

fig = plt.figure(num=1)
plt.subplot(212)
plt.plot(range(len(y_test)), y_test, color='b', label='Actual')
plt.plot(range(len(regr_pred)), regr_pred, color='r', label='Predicted')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.title('Comparison of Actual and Predicted Values')
plt.legend()

# 调整子图间距
plt.subplots_adjust(hspace=0.5)  # 增加垂直间距
plt.show()

index = X_test.index
prediction = pd.DataFrame(index = index)
prediction["GivenValue"] = y_test
prediction["XGBoost_Prediction"] = xgbr_pred
prediction


# In[ ]:





# In[ ]:





# In[ ]:




