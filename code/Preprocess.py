#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import func as dd

# 判斷是否要使用特徵選取，是的話將會變 true
check = False

def Preprocessing():
    # 讀資料集
    train = pd.read_csv('laptops_train.csv')
    test = pd.read_csv('laptops_test.csv')

    # 偵測是否有缺失值
    # print(train.isnull().sum() / len(train) * 100)
    # print(test.isnull().sum() / len(test) * 100)
    train = train.fillna("NaN")
    test = test.fillna("NaN")

    # 針對能視為是數值的特徵做處理
    def transNum(data):
        data["RAM"] = data["RAM"].str.replace("GB", "")
        data["RAM"] = data["RAM"].astype(float)
        data['Weight'] = data['Weight'].str.replace('kg','')
        data['Weight'] = data['Weight'].str.replace('s', '').astype(float)
        data["Screen Size"] = data['Screen Size'].str.replace('"','')
        data['Screen Size'] = data['Screen Size'].astype(float)
        return data

    train = transNum(train)
    test = transNum(test)
    #print(train.head())

    # Extract data
    # 統計 Screen 資訊，並提取資料成為新特徵 
    #print(train["Screen"].value_counts())
    def extract_screen_features(train):
        train = train.rename(columns = str.lower)
        train["resolution"] = train["screen"].str.extract(r'(\d+x\d+)')
        train["screentype"]= train["screen"].replace(r'(\d+x\d+)','', regex = True)
        train['screentype'] = train['screentype'].replace(r'(Full HD|Quad HD|Quad HD|\+|/|4K Ultra HD)','', regex = True)
        train['touchscreen'] = train['screentype'].str.extract(r'(Touchscreen)')
        train['screentype'] = train['screentype'].replace(r'(Touchscreen)','', regex = True)
        train['touchscreen'] = train['touchscreen'].replace('Touchscreen', 1)
        train['touchscreen'] = train['touchscreen'].replace(np.nan, 0)
        train['screentype'] = train['screentype'].replace(r' ','', regex = True)
        train['screentype'] = train['screentype'].replace(r'^\s*$', np.nan, regex = True)
        train = train.drop("screen",axis = 1)
        return train

    train = extract_screen_features(train)
    test = extract_screen_features(test)
    #print(train.head())

    def extract_from_cpu(train):
        train["freq"] = train["cpu"].str.extract(r'(\d+(?:\.\d+)?GHz)')
        train["freq"] = train["freq"].str.replace('GHz','')
        train["freq"] = train["freq"].astype(float)
        train["cpu"] = train["cpu"].str.replace(r'(\d+(?:\.\d+)?GHz)','',regex =True)
        return train
    
    train = extract_from_cpu(train)
    test = extract_from_cpu(test)
    train.head()

    def manufacturer(train):
        train['cpu_manftr'] = train['cpu'].str.extract(r'^(\w+)') 
        train['gpu_manftr'] = train['gpu'].str.extract(r'^(\w+)') 
        return train

    train = manufacturer(train)
    test = manufacturer(test)
    #print(train.head())

    # 統計記憶體是用 HDD 或 SSD等分類，並提取相關資訊
    #print(train[" storage"].value_counts())
    def extract_from_storage(train):
        train["storage1"] = train[" storage"]
        train['storage1'] = train['storage1'].str.replace('1.0TB','1TB', regex = True)
        train['storage1'] = train['storage1'].str.replace('1TB','1000GB')
        train['storage1'] = train['storage1'].str.replace('2TB','2000GB')
        train['storage1'] = train['storage1'].str.replace('GB','')
        train['storage2'] = train['storage1'].str.replace(r' ','')
        storage1 = []
        storage2 = []
        for i in train['storage2']:
            if len(re.findall(r'\+', i)) == 1: 
                # Double drive
                one = re.findall(r'(\w+)', i)
                storage1.append(one[0])
                storage2.append(one[1])
            else: 
                # Single drive
                one = re.findall(r'(\w+)', i)
                storage1.append(one[0])
                storage2.append('NaN')


        #extracting size and type of primary storage
        storage1size = []
        storage1type = []
        for i in storage1:
            storage1type.append(re.findall(r'(\D\w+)', i)[0])
            storage1size.append(re.findall(r'(\d+)', i)[0])


        #extracting size and type of secondary storage
        storage2size = []
        storage2type = []
        for i in storage2:
            if i != 'NaN':
                storage2type.append(re.findall(r'(\D\w+)',i)[0])
                storage2size.append(re.findall(r'(\d+)',i)[0])
            else:
                storage2type.append('NaN')
                storage2size.append(0)
        train['primarystorage_size'] = storage1size
        train['primarystorage_type'] = storage1type
        train['secondarystorage_size'] = storage2size
        train['secondarystorage_type'] = storage2type

        train["primarystorage_size"] = train["primarystorage_size"].astype(float)
        train["secondarystorage_size"] = train["secondarystorage_size"].astype(float)
        train = train.drop(['storage1','storage2',' storage'], axis = 1)
        train = train.replace({'NaN' : np.nan})
        return train
    
    train = extract_from_storage(train)
    test = extract_from_storage(test)
    #print(train.head())

    def Currency_change(train):
        Price = []
        for i in train['price']:
            value = i / 487.94
            Price.append(value)
        
        train['price'] = Price
        return train
    
    # 幣值轉換
    train = Currency_change(train)
    test = Currency_change(test)
    
    # 資料轉換
    dd.data_encoding(train, "train1.csv")
    dd.data_encoding(test, "test1.csv")
      
    # 使用特徵選取
    # correlation = dd.feature_Select(data)
    # dd.show_Result(correlation)
    # dd.output_feature_Select(data, 12, 'dataset.csv',correlation)
    # check = True
    
def checkUsingFS():
    if check:
        return True
    
Preprocessing()
print("Preprocess Done~~")






