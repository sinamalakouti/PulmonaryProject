#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:05:13 2020

@author: sina
"""

from flask import Flask
from flask import Flask,jsonify,request
from flask import Flask, request, jsonify, render_template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.externals import joblib

from flask import Flask
from flask import Flask,jsonify,request
from flask import Flask, request, jsonify, render_template
import io
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure



from sklearn.externals import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import RUSBoostClassifier
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import  FeatureUnion
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Imputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.metrics import classification_report
from sklearn  .metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, \
    precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn import tree, svm
import io
import csv
from scipy.stats import mode
from sklearn.metrics import f1_score
from flask import render_template
from bidi.algorithm import get_display
import arabic_reshaper

import matplotlib.pyplot as plot
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV




def read_dataset():
    global data
    # data = pd.read_csv("../dataset/SPSS_DATA.csv")
    data = pd.read_csv("../dataset/dataset3.csv")
    data = pd.read_csv("../dataset/weka_data.csv")
    # data = pd.read_csv("../dataset/weka_data2_small.csv")
    data2 = pd.read_csv("../dataset/dataset.csv")
    data = pd.concat([data, data2['Discharge_Status']],axis = 1)
    data.replace(to_replace = '?', value = np.nan,inplace = True)
    return data


def train_and_serialize_models():

    data = read_dataset()

    data.drop(['Age_cat.65yr'],1, inplace = True)
    data.drop(['All.kind.smoker'],1, inplace = True)

    feature_headers = list(data.columns.values)
    categorical_features = []
    numerical_features = []
    binary_features = []
    class_headers = []


    for att in feature_headers:

        if att == 'Atelectasis' or att == 'Discharge_Status':
            class_headers.append(att)
            
        elif data [att].nunique() < 2:
            data [att].drop([att], axis=1, inplace=True)
            
        else:
            if data [att].nunique()> 5:
                numerical_features.append(att)
                

            elif data [att].nunique() == 2:
                binary_features.append(att)
                

            else:
                categorical_features.append(att)

    fcat = categorical_features
    fnum = numerical_features
    fbin = binary_features


    stm = SMOTE(random_state = 42)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value='missing', strategy="median")),
        ('varianceThreshhold', VarianceThreshold(threshold=(.8 * (1 - .8)))),
        ('standardscaler', RobustScaler(quantile_range=(0.1, 0.9), with_centering=False))
    ])
    


    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value='missing', strategy="most_frequent")),
        ('varianceThreshhold', VarianceThreshold(threshold=(.8 * (1 - .8)))),
        # ('standardscaler', RobustScaler(quantile_range=(0.1, 0.9), with_centering=False))
    ])
    

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value='missing', strategy="most_frequent")),
        ('varianceThreshhold', VarianceThreshold(threshold=(.8 * (1 - .8)))),
        # ('onehot', OneHotEncoder(handle_unknown='ignore')),
        # ('standardscaler', RobustScaler(quantile_range=(0.1, 0.9), with_centering=False))
    ])

    preprocessor = ColumnTransformer (
            transformers=[
                    ('num', numeric_transformer, fnum),
                    ('cat', categorical_transformer, fcat),
                    ('bin', binary_transformer, fbin)
                    ])

    pp = [
        ("preprocessing", preprocessor)]

    classifiers_lables = ['ada', 'SMOTEBoost', "RUSBoost", 'brfc', 'SVMBoost', "rf","dt"]

    classifier_pool = {
                       "ada": AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=1), n_estimators=120, learning_rate=1),
                       "SMOTEBoost": AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=1), n_estimators=140, learning_rate=1),
                       "RUSBoost": RUSBoostClassifier(n_estimators=300, learning_rate=0.1),   
                       "brfc": BalancedRandomForestClassifier(n_estimators=200, max_depth=1, random_state=24),
                       "SVMBoost": AdaBoostClassifier(base_estimator=svm.SVC( probability = True, kernel='linear'), n_estimators=20, learning_rate=0.9),
                       "rf": RandomForestClassifier(n_estimators=200, max_depth = 1, random_state=24),
                       "dt": LogisticRegression()                
                       }


    data.Atelectasis.fillna(data['Atelectasis'].mode()[0], inplace=True)
    Y = pd.concat([data['Atelectasis']], axis=1)
    X = data.drop(['Atelectasis', 'Discharge_Status'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state =2)
    data_test = pd.concat([X_test, y_test], axis = 1)
    data_test.to_csv("../dataset/final_test.csv")


    for clf_label in classifiers_lables:
        print("training  ", clf_label)

        estimator = []

        if clf_label == 'SMOTEBoost' or clf_label == 'SVMBoost':

            estimator = [
            ("preprocessing", preprocessor),
            ('stmt', stm),
            ('classifiers', Pipeline([(clf_label, classifier_pool[clf_label])]))]
        else:

            estimator = [
            ("preprocessing", preprocessor),
            ('classifiers', Pipeline([(clf_label, classifier_pool[clf_label])]))]

        pp = Pipeline(estimator)
        
      
        pp.fit(X_train, y_train)
        y_pred = pp.predict(X_test)

        
        model_name = clf_label + ".pkl"
        joblib.dump(pp, model_name)

    vote_learner = VotingClassifier (estimators = [
		('ada', classifier_pool['ada']),
		('SMOTEBoost', classifier_pool['SMOTEBoost']),
		('RUSBoost', classifier_pool['RUSBoost']),
		('brfc', classifier_pool['brfc']),
		('SVMBoost', classifier_pool['SVMBoost'])
		])
       

app = Flask(__name__)

def plot_final(x, y, xlabel, ylabel, file_address_name,title ):
	xlbl = get_display( arabic_reshaper.reshape(xlabel))
	ylbl = get_display( arabic_reshaper.reshape(ylabel))
	title = get_display( arabic_reshaper.reshape(title))
	x_true = []

	for item in x:
		x_true.append(get_display(arabic_reshaper.reshape(item)))

	fig = Figure(figsize= (18,14))
	axis = fig.add_subplot(1, 1, 1)
	axis.set_title(title, fontsize=22)
	axis.set_xlabel(xlbl,fontsize=22)
	axis.set_ylabel(ylbl,fontsize=22)


	for tick in axis.get_xticklabels():
		tick.set_rotation(45)
		tick.set_fontsize(20)

	axis.bar(x_true,y)
	pngImage = io.BytesIO()
	FigureCanvas(fig).print_png(pngImage)

	pngImageB64String = "data:image/png;base64,"
	pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
 
	return pngImageB64String


def load_models():
	

	classifiers_lables = ['ada', 'SMOTEBoost', "RUSBoost", 'brfc', 'SVMBoost']
	classifiers = {}
	pred_ada = []
	pred_ada = []

	for clf_label in classifiers_lables:

		model_file_name = clf_label + ".pkl"
		classifiers [clf_label] = joblib.load(model_file_name)


	
	return classifiers





@app.route('/')
def form():

	return render_template("index.html")

@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    test_data = pd.read_csv(stream)
    # test_data = pd.read_csv("../dataset/final_test.csv")
    # test_data.drop(['Age_cat.65yr'],1, inplace = True)
    # test_data.drop(['All.kind.smoker'],1, inplace = True)
    test_data.replace(to_replace = '?', value = np.nan,inplace = True)

    y_test = pd.concat([test_data['Atelectasis']], axis=1)
    X_test = test_data.drop(['Atelectasis'], axis=1)
    classifiers = load_models()
    vote_learner = VotingClassifier (estimators = [
		('ada', classifiers['ada']),
		('SMOTEBoost', classifiers['SMOTEBoost']),
		('RUSBoost', classifiers['RUSBoost']),
		('brfc', classifiers['brfc']),
		('SVMBoost', classifiers['SVMBoost'])
		])

    prediction_matrix = np.zeros((y_test.shape[0], 5))
    prediction_matrix[:,0] = classifiers['ada'].predict(X_test)
    prediction_matrix[:,1] = classifiers['SMOTEBoost'].predict(X_test)
    prediction_matrix[:,2] = classifiers['RUSBoost'].predict(X_test)
    prediction_matrix[:,3] = classifiers['brfc'].predict(X_test)
    prediction_matrix[:,4] = classifiers['SVMBoost'].predict(X_test)
    vote_result = mode(prediction_matrix, axis=-1)[0]
    print("auc vote  ", roc_auc_score(y_test, vote_result))
    print("auc ada  ", roc_auc_score(y_test, prediction_matrix[:,0]))
    print("auc smote  ", roc_auc_score(y_test, prediction_matrix[:,1]))
    print("auc rus  ", roc_auc_score(y_test, prediction_matrix[:,2]))
    print("auc rbfc  ", roc_auc_score(y_test, prediction_matrix[:,3]))
    print("auc svm  ", roc_auc_score(y_test, prediction_matrix[:,4]))
    model_names_english = ['ada', 'SMOTEBoost', 'RUSBoost', 'brfc', 'SVMBoost' ]
    model_names_persian = ['آدابوست','اسموت‌بوست','آریواس بوست','اس‌وی‌ام بوست','جنگل تصادفی متعادل', 'رای‌گیری']
    auc_scores = [0.0,0.0,0.0,0.0,0.0,0.0]
    f_scores = [0.0,0.0,0.0,0.0,0.0,0.0]

    counter = 0 
    for clf_label in model_names_english:
    	    prediction_matrix[:,counter] = classifiers[clf_label].predict(X_test)
    	    auc_scores[counter] = roc_auc_score(y_test, prediction_matrix[:,counter])
    	    f_scores[counter] = f1_score(y_test, prediction_matrix[:,counter], average='weighted')
    	    counter +=1

    f_scores[5] = f1_score(y_test, vote_result, average='weighted')
    auc_scores[5] = roc_auc_score(y_test, vote_result, average='weighted')
    print(f_scores)

    auc_img = plot_final(model_names_persian, auc_scores, 'نام مدل‌ها', "AUC", "AUC_finaltest", "مقایسه نتایج بر اساس AUC" )
    f_img = plot_final(model_names_persian, f_scores, 'نام مدل‌ها', "نمره اف۱","F1_finaltest" , "مقایسه نتایج بر اساس نمره اف۱")
    print(vote_result)



    return render_template("result.html", url_auc =auc_img, url_f1 = f_img, 
    	p1 =vote_result[0] ,p2=vote_result[1] ,p3= vote_result[2] ,p4= vote_result[3] ,p5= vote_result[4]
    	 ,p6 = vote_result[5] ,p7=vote_result[6] ,p8=vote_result[7] ,p9=vote_result[8] ,p10= vote_result[9] )





train_and_serialize_models()


