#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 資料轉換
def data_encoding(data, new_data):
    # 進行 Label Encoding
    for column in data.columns:
        if data[column].dtype == 'object' and column != 'Price':
            label_encoder = LabelEncoder()
            data[column] = label_encoder.fit_transform(data[column])
    
    # 進行 One-Hot Encoding
    encoded_data = pd.get_dummies(data, drop_first = True)
    # 保存成CSV檔案
    encoded_data.to_csv(new_data, index=False, header=True)
    return new_data

def feature_Select(data):
    correlation_matrix = data.corr()
    correlation_with_target = correlation_matrix['price'].abs()
    sorted_correlation = correlation_with_target.sort_values(ascending=False)

    print(sorted_correlation)
    return sorted_correlation

def show_Result(correlation):
    # 提取名稱和相關係數值
    fields = correlation.index
    correlation_values = correlation.values

    # 建立長條圖
    plt.bar(fields, correlation_values)
    plt.xlabel('Fields')
    plt.ylabel('Correlation')
    plt.title('Bar Chart of Pearson Correlation')

    # 旋轉名稱
    plt.xticks(rotation=90)
    plt.show()

# 輸出檔案(原始資料集、要取出的特徵數量(有包括Price)、輸出的檔案名稱)
def output_feature_Select(data,n,csv_name,correlation):
    # 儲存原始欄位名稱順序
    original_columns_order = data.columns.tolist()
    # 取出前n個相關係數的欄位名稱
    top_n_features = correlation[:n].index.tolist()

    # 按照原始欄位順序重新排序欄位名稱
    sorted_features = [column for column in original_columns_order if column in top_n_features]

    # 提取相應欄位的資料
    extracted_data = data[sorted_features]

    # 輸出為新的CSV檔案，並保持原始欄位順序
    extracted_data.to_csv(csv_name, index=False, columns=sorted_features)


# In[ ]:




