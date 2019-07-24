import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler# reading data
from sklearn.ensemble import RandomForestClassifier
def read_dataset():
    global data
    data = pd.read_csv("../dataset/Dataset.csv")

def proc_corton():
        global data
        data.Corton.fillna(data['Corton'].mode()[0], inplace=True)
        df_dummies = pd.get_dummies(data['Corton'], prefix='Corton')
        data = pd.concat([data, df_dummies], axis=1)
        data.drop(['Corton'], axis=1, inplace =True)
def proc_firstDiagnosis():
    global data
    data.First_Diagnosis.fillna(data['First_Diagnosis'].mode()[0], inplace=True)
    df_dummies = pd.get_dummies(data['First_Diagnosis'], prefix='First_Diagnosis')
    data = pd.concat([data, df_dummies], axis=1)
    data.drop(['First_Diagnosis'], axis=1, inplace=True)

def proc_kedney():
    global data
    data.Kedney.fillna(data['Kedney'].mode()[0], inplace=True)
    df_dummies = pd.get_dummies(data['Kedney'], prefix='Kedney')
    data = pd.concat([data, df_dummies], axis=1)
    data.drop(['Kedney'], axis=1, inplace =True)
    
def proc_Diabetes():
    global data
    data.Diabetes.fillna(data['Diabetes'].mode()[0], inplace=True)
    df_dummies = pd.get_dummies(data['Diabetes'], prefix='Diabetes')
    data = pd.concat([data, df_dummies], axis=1)
    data.drop(['Diabetes'], axis=1, inplace =True)
def proc_HighBlood():
    global data
    data.HighBlood.fillna(data['HighBlood'].mode()[0], inplace=True)
    df_dummies = pd.get_dummies(data['HighBlood'], prefix='HighBlood')
    data = pd.concat([data, df_dummies], axis=1)
    data.drop(['HighBlood'], axis=1, inplace =True)
def proc_KIND_SSurg():
    global data
    data.KIND_SSurg.fillna(data['KIND_SSurg'].mode()[0], inplace= True)
    df_dummies = pd.get_dummies(data['KIND_SSurg'], prefix='KIND_SSurg')
    data = pd.concat([data, df_dummies], axis=1)
    data.drop(['KIND_SSurg'], axis=1, inplace=True)
def proc_FEV1():
    data.FEV1.fillna(data['FEV1'].mean(), inplace= True)
def proc_FVC():
    data.FVC.fillna(data['FVC'].mean(), inplace= True)
def proc_FEV1FVC():
    data.FEV1FVC.fillna(int(data['FEV1FVC'].mean()), inplace= True)
def proc_PEFR():
    data.PEFR.fillna(data['PEFR'].mean(), inplace= True)
def proc_MMEF():
    data.MMEF.fillna(data['MMEF'].mean(), inplace= True)
def proc_FEV1_A():
    data.FEV1_A.fillna(int(data['FEV1_A'].mean()), inplace= True)
def proc_FVC_A():
    data.FVC_A.fillna(int(data['FVC_A'].mean()), inplace= True)
def proc_FEV1FVC_A():
    data.FEV1FVC_A.fillna(int(data['FEV1FVC_A'].mean()), inplace= True)
def proc_PEFR_A():
    data.PEFR_A.fillna(int(data['PEFR_A'].mean()), inplace= True)
def proc_MMEF_A():
    data.MMEF_A.fillna(int(data['MMEF_A'].mean()), inplace= True)
def proc_FBS():
    data.FBS.fillna(int(data['FBS'].mean()), inplace= True)
# regression!
def proc_TSH():
    data.TSH.fillna(data['TSH'].mean(), inplace= True)
def proc_UricAsid():
    data.UricAsid.fillna(data['UricAsid'].mean(), inplace= True)
def proc_Creatinin():
    data.Creatinin.fillna(data['Creatinin'].mean(), inplace= True)
def proc_ABGAIR():
    data.ABGAIR.fillna(data['ABGAIR'].mode()[0], inplace= True)
def proc_PH_A():
    data.PH_A.fillna(data['PH_A'].mean(), inplace= True)
def proc_PCO2_A():
    data.PCO2_A.fillna(int(data['PCO2_A'].mean()), inplace= True)
def proc_HCO3_A():
    data.HCO3_A.fillna(int(data['HCO3_A'].mean()), inplace= True)
def proc_PO2_A():
    data.PO2_A.fillna(int(data['PO2_A'].mean()), inplace= True)
def proc_O2Sat_A():
    data.O2Sat_A.fillna(int(data['O2Sat_A'].mean()), inplace= True)
def proc_O2Sat_B():
    data.O2Sat_B.fillna(int(data['O2Sat_B'].mean()), inplace= True)
def proc_ClampTIME():
    data.ClampTIME.fillna(int(data['ClampTIME'].mean()), inplace= True)
def proc_PUMPTIME():
    data.PUMPTIME.fillna(int(data['PUMPTIME'].mean()), inplace= True)
def proc_SKINTOSKINTIME():
    data.SKINTOSKINTIME.fillna(int(data['SKINTOSKINTIME'].mean()), inplace= True)
def proc_MVTIMEh():
    data.MVTIMEh.fillna(int(data['MVTIMEh'].mean()), inplace= True)
def proc_BLOODT():
    data.BLOODT.fillna(data['BLOODT'].mode()[0], inplace= True)
#TODO : handling class missing value
def proc_Atelectasis():
    data.Atelectasis.fillna(data['Atelectasis'].mode()[0], inplace= True)
    #todo
def proc_SPAP():
    data.drop(['SPAP'],1, inplace=True)

def normalizing(X_train, X_test):
    mean = X_train.mean()
    std = X_train.std()
    train_norm = (X_train - mean) / std
    test_norm =  (X_test - mean ) / std
    print(test_norm)
    # scaler = preprocessing.StandardScaler().fit(X_train)
    # train_nor
    return[train_norm, test_norm]
def test_train_splitting(test_fraction, k, isFeatureSelection, class_label, random_state = 2.2):
    y = Y[class_label].values[:]
    # print(y)
    X = data.values[:,1:]
    print(X)

    if ( isFeatureSelection == True):
        X = kBest(k,X,y)
        print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size= test_fraction, random_state=2)
    return [X_train, X_test, y_train, y_test]
def train():
    [X_train, X_test, y_train, y_test] = test_train_splitting(0.2,54,True,  'Atelectasis')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
     # print("mean is \t" , X_train[1].mean)
    classifiers = [ ('tree',tree.DecisionTreeClassifier(random_state = 2, max_depth = 10)),
      ('adaboost',AdaBoostClassifier(n_estimators=2040, learning_rate=0.8, random_state=2)),
      ('randomforest', RandomForestClassifier(n_estimators=100, max_depth=5,random_state=2)),
      ('svm', svm.LinearSVC())]
    for (clf_label, clf) in classifiers: 
         print(clf_label)
         print(X_train.shape)
         clf.fit(X_train, y_train)
         y_pred = clf.predict(X_test)
         print(classification_report(y_test, y_pred, ))
         fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred , pos_label=1)
         print("AUC is ", metrics.auc(fpr, tpr))
         print("fscore   " , f1_score(y_test, y_pred, average="weighted"))
         print(y_pred)
        #  print("model  ", clf_label, "  score :   ", clf.score(X_test,y_test))
    #  print("true y is ")
    #  print(y_test)
def kfold (k,X,y):
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, y)
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # [X_train, X_test] = normalizing(X_train, X_test)
        scaler = preprocessing.StandardScaler().fit(X_train)
        scaler.fit_transform(X_train)
        scaler.fit_transform(X_test)
         # print("mean is \t" , X_train[1].mean)
        classifiers = [ ('tree',tree.DecisionTreeClassifier()),
        ('adaboost',AdaBoostClassifier(n_estimators=100)),
        ('svm', svm.LinearSVC())]
        for (clf_label, clf) in classifiers:
            print(clf_label)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred , pos_label=1)
            print("AUC is ", metrics.auc(fpr, tpr))


        
def kBest(k, X , y):
        X_new = SelectKBest(f_regression, k).fit_transform(X, y) 
        return X_new  
def main ():
    global data
    global Y
    read_dataset()
    preprocessing()
    Y  = pd.concat([data['Atelectasis'], data['Discharge_Status']], axis=1)
    data.drop(['Discharge_Status','Atelectasis'],1, inplace = True)
    train()
def preprocessing():
    # dropping all the columns(features) with only one unique value
    global data
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
    proc_FEV1()
    proc_FVC()
    proc_FEV1FVC()
    proc_PEFR()
    proc_MMEF()
    proc_FEV1_A()
    proc_FVC_A()
    proc_FEV1FVC_A()
    proc_PEFR_A()
    proc_MMEF_A()
    proc_FBS()
    proc_TSH()
    proc_UricAsid()
    proc_Creatinin()
    proc_ABGAIR()
    proc_PH_A()
    proc_PCO2_A()
    proc_HCO3_A()
    proc_PO2_A()
    proc_O2Sat_A()
    proc_O2Sat_B()
    proc_ClampTIME()
    proc_PUMPTIME()
    proc_SKINTOSKINTIME()
    proc_MVTIMEh()
    proc_BLOODT()
    proc_Atelectasis()
    proc_SPAP()
    
if __name__ == "__main__":
    main()