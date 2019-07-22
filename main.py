import pandas as pd
import numpy as np

def read_dataset():
    global data
    data = pd.read_csv("../dataset/Dataset.csv")
def proc_corton():
        data.Corton.fillna(data['Corton'].mode()[0], inplace=True)
def proc_firstDiagnosis():
    print(data.groupby('First_Diagnosis').count())
    print(data['First_Diagnosis'].mode()[0])
    data.First_Diagnosis.fillna(data['First_Diagnosis'].mode()[0], inplace=True)

def proc_kedney():
    data.Kedney.fillna(data['Kedney'].mode()[0], inplace=True)
    
def proc_Diabetes():
    data.Diabetes.fillna(data['Diabetes'].mode()[0], inplace=True)
def proc_HighBlood():
    data.HighBlood.fillna(data['HighBlood'].mode()[0], inplace=True)
def proc_KIND_SSurg():
    data.KIND_SSurg.fillna(data['KIND_SSurg'].mode()[0], inplace= True)

def main ():
    read_dataset()
    preprocessing()
    print(data.isna().sum())

        
    # print(data.nunique())
    # for val in data['SPAP']:
    #     if(pd.isna(val)):
    #     #     print(c)
    #     #     print(val)
    #     # else:
    #     #     print(c)
    #     #     print(val)
    #     c = c + 1

def preprocessing():
    # dropping all the columns(features) with only one unique value
    features_headers = list(data.columns.values)
    print("before dropping  ", len(features_headers))
    for att in features_headers:
        if(data.nunique()[att] < 2):
            data.drop([att],axis=1,inplace=True)
    print("after dropping   ", len(list(data.columns.values)) )
    proc_corton()
    proc_firstDiagnosis()
    proc_kedney()
    proc_Diabetes()
    proc_HighBlood()
    proc_KIND_SSurg()

if __name__ == "__main__":
    main()