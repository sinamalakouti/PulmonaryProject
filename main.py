import pandas as pd
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import classification_report
from sklearn  .metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, \
    precision_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import  FeatureUnion
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Imputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import Axes3D
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import RUSBoostClassifier
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from flask import Flask
from sklearn.externals import joblib
from bidi.algorithm import get_display
import arabic_reshaper
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

    

# from maatpy.classifiers import SMOTEBoost



def outliers_univariate():
    pass

    


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



def normalizing(X_train, X_test):
    mean = X_train.mean()
    std = X_train.std()
    train_norm = (X_train - mean) / std

    test_norm = (X_test - mean) / std
    print(test_norm)
    # scaler = preprocessing.StandardScaler().fit(X_train)
    # train_nor
    return [train_norm, test_norm]


# define grid search params for each model 
def gridSearch_params():
    svm_param = {

        'feature_selection__bestK__k': (3, 10, 20, 30, 40),
      #  'feature_selection__pca__n_components': (3, 10, 20, 30, 40),
        'classifiers__svm__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        # 'classifiers__svm__degree':(3,5,7,9,11,13,20)
        
    }
    tree_param = {
        # 'feature_selection__bestK__k': (3, 10, 20, 30 , 40),
     #   'feature_selection__pca__n_components': (3, 10, 20, 30, 40),
        'classifiers__tree__criterion': ('gini', 'entropy'),
        'classifiers__tree__max_depth': (5, 10, 15, 20, 25, 30)}
    adaboost_params = {
        # 'feature_selection__bestK__k': (3, 10, 20, 30 , 40, ),
       # 'feature_selection__pca__n_components': (3, 10, 20, 30, 40),
        'classifiers__ada__n_estimators': (10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        'classifiers__ada__learning_rate': (0.1, 0.3, 0.5, 0.7, 0.9, 1)}

    smote_params = {
        # 'feature_selection__bestK__k': (3, 10, 20, 30 , 40, ),
       # 'feature_selection__pca__n_components': (3, 10, 20, 30, 40),
        'classifiers__smote__n_estimators': (10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        'classifiers__smote__learning_rate': (0.1, 0.3, 0.5, 0.7, 0.9, 1)}

    svmBoost_params = {
        # 'feature_selection__bestK__k': (3, 10, 20, 30 , 40, ),
       # 'feature_selection__pca__n_components': (3, 10, 20, 30, 40),
        'classifiers__svmBoost__n_estimators': (10, 20, 30, 40, 50),
        'classifiers__svmBoost__learning_rate': (0.1, 0.3, 0.5, 0.9, 1)}
    brfc_params = {
        'classifiers__brfc__n_estimators': ( 10, 20, 30, 40, 50, 60, 70, 70, 90, 100,200,300),
        'classifiers__brfc__max_depth': ( 1, 5, 10)

    }
    rus_params = {
        # 'feature_selection__bestK__k': (3, 10, 20, 30 , 40, ),
       # 'feature_selection__pca__n_components': (3, 10, 20, 30, 40),
        'classifiers__rus__n_estimators': (10, 20, 30, 40, 50, 60, 70, 80, 90, 100,  200, 300),
        'classifiers__rus__learning_rate': (0.1, 0.3, 0.5, 0.7, 0.9, 1)}
    logistic_params = {
        'feature_selection__bestK__k': (3, 10, 20, 40),
       # 'feature_selection__pca__n_components': (3, 10, 20, 30, 40),
    }
    params = {'tree': tree_param, 'ada': adaboost_params, 'logistic': logistic_params, 'svm': svm_param, 'rus': rus_params, 'brfc': brfc_params, 'svmBoost': svmBoost_params, 'smote': smote_params}
    return params


# splitting data set to the train/test sets 
def test_train_splitting(test_fraction, k, isFeatureSelection, class_label, random_state=2):
    y = Y[class_label].values[:]

    X = data.values[:, 1:]

    if (isFeatureSelection == True):
        X = kBest(k, X, y)
        print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=random_state)
    return [X_train, X_test, y_train, y_test]

# training the model
def train():
    test_with_feature_selection()
    # test_without_feature_selection()


def test_without_feature_selection():
    classifiers = [
        ('adaboost', AdaBoostClassifier(n_estimators=2040, learning_rate=0.8))]
    parameters = gridSearch_params()
    for (clf_label, clf) in classifiers:

        [X_train, X_test, y_train, y_test] = test_train_splitting(0.2, 0, False, 'Atelectasis')
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        print(clf_label)
        model_param = parameters[clf_label]
        skf = StratifiedKFold(n_splits=3)
        pca = PCA()
        pipe = Pipeline(steps=[('pca', pca), (clf_label, clf)])
        gridModel = GridSearchCV(pipe, param_grid=model_param, scoring='roc_auc', cv=skf)
        gridModel.fit(X_train, y_train)
        # p = gridModel.predict(X_test)
        # print(gridModel.cv_results_)
        # clf.fit(X_train, y_train)
        pred = gridModel.best_estimator_.predict(X_test)
        rep = classification_report(y_test, pred)
        print(rep)
        # print("AUC is ", gridModel.best_estimator_.score(X_test, y_test ))
        # print("accuracy is  ", accuracy_score(y_test, pred))
        from sklearn.metrics import roc_auc_score
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
        TN, FP, FN, TP = confusion_matrix(y_test, pred).ravel()
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        ACC = (TP + TN) / (TP + FP + FN + TN)
        print("ACCC\t", ACC)
        print("sensitivity\t", TPR)
        print("Specificity\t", TNR)
        mean_test_score = gridModel.cv_results_['mean_test_score']
        # for adaboost
        estimators = (gridSearch_params()['adaboost']['n_estimators'])
        print(estimators)
        learning_rates = list(gridSearch_params()['adaboost']['learning_rate'])
        n_learningRates = len(learning_rates)
        n_estimators = len(estimators)
    
        plt.xlabel("number of estimators")
        plt.ylabel("AUC")
        for i in range(0, n_learningRates - 1):
            plt.plot(estimators, mean_test_score[(i * n_estimators): ((i + 1) * n_estimators)],
                     label=str(learning_rates[i]) + " learning rate")
        plt.legend(loc='upper right')
        plt.show()
        print(gridModel.cv_results_)


def test_with_feature_selection():
    classifiers = [('tree', tree.DecisionTreeClassifier(max_depth=10, random_state=2)),
                   ('adaboost', AdaBoostClassifier(n_estimators=2040, learning_rate=0.8))]
    parameters = gridSearch_params()
    for (clf_label, clf) in classifiers:
        number_of_features = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        aucs = np.zeros(10)
        sensitivites = np.zeros(10)
        specifities = np.zeros(10)
        supports = np.zeros((1, 10))
        c = 0
        for n_feature in number_of_features:
            [X_train, X_test, y_train, y_test] = test_train_splitting(0.2, n_feature, True, 'Atelectasis')
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            print(clf_label)
            model_param = parameters[clf_label]

            skf = StratifiedKFold(n_splits=3)
            gridModel = GridSearchCV(clf, param_grid=model_param, scoring='roc_auc', cv=skf)
            gridModel.fit(X_train, y_train)
            
            pred = gridModel.best_estimator_.predict(X_test)
            rep = classification_report(y_test, pred)
            print(rep)
            # print("AUC is ", gridModel.best_estimator_.score(X_test, y_test ))
            # print("accuracy is  ", accuracy_score(y_test, pred))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
            TN, FP, FN, TP = confusion_matrix(y_test, pred).ravel()
            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP / (TP + FN)
            # Specificity or true negative rate
            TNR = TN / (TN + FP)
            ACC = (TP + TN) / (TP + FP + FN + TN)
            print("accuracy\t", ACC)
            print("sensitivity\t", TPR)
            print("Specificity\t", TNR)
            aucs[c] = gridModel.score(X_test, y_test)
            sensitivites[c] = TPR
            specifities[c] = TNR
            c = c + 1

        plt.plot(number_of_features, aucs)
        plt.xlabel("number of features")
        plt.ylabel("AUC score")
        plt.show()
        plt.plot(number_of_features, sensitivites)
        plt.xlabel("number of features")
        plt.ylabel("Sensitivity score")
        plt.show()
        plt.plot(number_of_features, specifities)
        plt.xlabel("number of features")
        plt.ylabel("Specificity score")
        plt.show()



def kfold(k, X, y):
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
        classifiers = [('tree', tree.DecisionTreeClassifier()),
                       ('adaboost', AdaBoostClassifier(n_estimators=10)),
                       ('svm', svm.LinearSVC())]
        for (clf_label, clf) in classifiers:
            print(clf_label)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
            print("AUC is ", metrics.auc(fpr, tpr))


def kBest(k, X, y):
    X_new = SelectKBest(f_regression, k).fit_transform(X, y)
    return X_new


def main():
    global data
    global Y
    read_dataset()
    data = data.dropna(axis='rows')

    print(data._get_numeric_data().columns)
    features_headers = list(data.columns.values)
    print(features_headers)
    preprocessing()
    Y  = pd.concat([data['Atelectasis'], data['Discharge_Status']], axis=1)
    data.drop(['Discharge_Status','Atelectasis'],1, inplace = True)
    selector = VarianceThreshold(threshold = 0.8)
    selector.fit(data)
    train()

 

def main_proc():
    data = read_dataset()
    # data.drop(['SPAP'], 1, inplace=True)
    # data.drop(['Age'],1,inplace = True)
    # data.drop(['Length_hospitalization'], 1, inplace = True)
    data.drop(['Age_cat.65yr'],1, inplace = True)
    data.drop(['All.kind.smoker'],1, inplace = True)
    
    feature_headers = list(data.columns.values)
    categorical_features = []
    numerical_features = []
    binary_features = []
    class_headers = []
    for att in feature_headers:
        print(att)
        if att == 'Atelectasis' or att == 'Discharge_Status':
            class_headers.append(att)
        elif  data [att].nunique() < 2:
            data.drop([att], axis=1, inplace=True)
        else:
            if ( data [att].nunique() > 5):
                numerical_features.append(att)
            elif data [att].nunique() == 2:
                binary_features.append(att)
            else:
                categorical_features.append(att)

    classifier_pool = {
                       # 'svm': svm.SVC(kernel = "rbf", gamma = 'scale', class_weight = 'balanced'),
                       # "ada": AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=1), n_estimators=10, learning_rate=0.1),
                       "smote": AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=1), n_estimators=10, learning_rate=0.1),
                        "brfc": BalancedRandomForestClassifier(n_estimators=10, max_depth=1, random_state=24),
                       "rus": RUSBoostClassifier(n_estimators=10, learning_rate=0.1),   
                       # "svmBoost": AdaBoostClassifier(base_estimator=svm.SVC( probability = True, kernel='linear'), n_estimators=10, learning_rate=0.1)
                       # "tree": tree.DecisionTreeClassifier(max_depth=10),
                       # "logistic": LogisticRegression(n_jobs=100, solver = "lbfgs", class_weight="balanced")
                       }
    data.Atelectasis.fillna(data['Atelectasis'].mode()[0], inplace=True)
    Y = pd.concat([data['Atelectasis'], data['Discharge_Status']], axis=1)
    X = data.drop(['Atelectasis', 'Discharge_Status'], axis=1)

    print(type(X))
    print(type(Y))  
    X_train, X_test, y_train, y_test = train_test_split(X, Y['Atelectasis'], test_size=0.3, random_state=2)
    for (clf_label) in classifier_pool:
        fit_test(X_train, y_train, X_test, y_test, categorical_features, numerical_features, binary_features,
                 class_headers, clf_label, classifier_pool)



def fit_test(X_train, y_train, X_test, y_test, fcat, fnum, fbin, class_headers, clf_label, classifiers):
    
    stm = SMOTE(random_state = 42)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value='missing', strategy="median")),
        ('varianceThreshhold', VarianceThreshold(threshold=(.8 * (1 - .8)))),
        ('standardscaler', RobustScaler(quantile_range=(0.1, 0.9)))
    ])

    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value='missing', strategy='most_frequent')),
        ('varianceThreshhold', VarianceThreshold(threshold=0.2)),

        ('standardscaler', RobustScaler(quantile_range=(0.1, 0.9), ))
        ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value='missing', strategy='most_frequent')),
        ('varianceThreshhold', VarianceThreshold(threshold=0.2)),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('standardscaler', RobustScaler(quantile_range=(0.1, 0.9), with_centering=False))
        ])

    # class_transformer = Pipeline(steps= [('imputer', Imputer(missing_values="NaN", strategy = 'most_freq', axis = 1))])
    preprocessor1 = ColumnTransformer(transformers=[
        ('num', numeric_transformer, fnum),
        ('cat', categorical_transformer, fcat),
        ('bin', binary_transformer, fbin)
        ])

    estimator = [
        ("preprocessing", preprocessor1),

        # ("feature_selection", FeatureUnion([
        #     ('bestK', SelectKBest(k=36)),
        #    # ('pca', PCA(n_components=10, random_`te=2))
        # ])),
        # ('stmt', stm),
        ('classifiers', Pipeline([(clf_label, classifiers[clf_label])]))]
    # return estimator

    if clf_label == 'smote' or clf_label == 'svmBoost':
        estimator = [
        ("preprocessing", preprocessor1),

        # ("feature_selection", FeatureUnion([
        #     ('bestK', SelectKBest(k=36)),
        #    # ('pca', PCA(n_components=10, random_state=2))
        # ])),
        ('stmt', stm),
        ('classifiers', Pipeline([(clf_label, classifiers[clf_label])]))]

    parameters = {
    }

    pp = Pipeline(estimator)

    grid_search = GridSearchCV(pp, param_grid=gridSearch_params()[clf_label], cv=10, scoring='normalized_mutual_info_score', refit= True)

    print(X_test.shape)
    print(y_test.shape)
    grid_search.fit(X_train, y_train)
    model_name = clf_label + ".pkl"
    # joblib.dump(pp, model_name) x

    print("model is \t", clf_label)
  
    test =  pd.concat([X_test, y_test], axis=1)
    #test.dropna(axis = 0, inplace = True)
   
    y_test = test['Atelectasis']
    X_test = test.drop(['Atelectasis'], axis = 1)
    X_cols_name = list(X_test.columns.values) 
    # sm = SMOTE (random_state=24)

    X_test = pd.DataFrame(X_test, columns = X_cols_name)
    y_test = pd.DataFrame(y_test)
#
    print("best estimator param  is is \t" , grid_search.best_params_ )
    print("best score is \t", grid_search.best_score_)

    print("grid search score is \t", grid_search.score(X_test, y_test))
    pred = grid_search.best_estimator_.predict(X_test)
    rep = classification_report(y_test, pred)
   
    print("F1socre of gird_search is \t", grid_search.score(X_test, y_test))

    print("roc_auc_score is  ", roc_auc_score(y_test, pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
    TN, FP, FN, TP = confusion_matrix(y_test, pred).ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    ACC = (TP + TN) / (TP + FP + FN + TN)
    print("Accuracy\t", ACC)
    print("sensitivity\t", TPR)
    print("Specificity\t", TNR)
    print("real fsocre    ", f1_score(y_test, pred))

    x_ada= [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    x_smote = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    x_rus = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    x_rfcb = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    x_svmboost =[10,20,30,40]

    y_ada = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    y_smote = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    y_rus = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    y_rfcb = [1, 5, 10]
    y_svmboost =[0.1,0.3,0.5,0.9,1]

    values = grid_search.cv_results_['mean_test_score']
    x = []
    y =[]
    xlabel =''
    ylabel = ''
    label = ''
    address_save =  clf_label
    
    if clf_label == 'ada':
        x = x_ada
        y = y_ada
        xlabel ='تعداد دسته‌بند‌ها'
        ylabel = 'نمره اف ۱'
        label = 'نرخ یادگیری'
        

    elif clf_label == 'smote':
        x = x_smote
        y = y_smote

        xlabel ='تعداد دسته‌بند‌ها'
        ylabel = 'نمره اف ۱'
        label = ' نرخ یادگیری'
    elif clf_label == 'rus':
        x = x_rus
        y = y_rus

        xlabel ='تعداد دسته‌بند‌ها'
        ylabel = 'نمره اف ۱'
        label = 'نرخ یادگیری'
    elif clf_label == 'brfc':
        x = x_rfcb
        y = y_rfcb

        xlabel ='دعداد دسته‌بند‌ها'
        ylabel = 'نمره اف ۱'
        label = 'عمق بیشینه'
    elif clf_label == 'svmBoost':
        x = x_svmboost
        y = y_svmboost

        xlabel ='تعداد دسته‌بند‌ها'
        ylabel = 'نمره اف ۱'
        label = 'نرخ یادگیری'

    plot_gridSearch(x, y, values, xlabel, ylabel, label,address_save)



def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

def plot_gridSearch(x1, x2, values, xlabel, ylabel, label,address_save):

    font_labels = {'family': 'B Nazanin'}
    
    for i in range(0,len(x2)):
        str1 =  label + ":  " + str(x2[i])
        strlbl = get_display( arabic_reshaper.reshape(str1))
        xlbl = get_display( arabic_reshaper.reshape(xlabel))
        ylbl = get_display( arabic_reshaper.reshape(ylabel))    
        plt.plot(x1, values[i * len(x1) : (i + 1) * len(x1) ], label = strlbl)
        
        plt.xlabel(xlbl, fontdict = font_labels)
        plt.ylabel(ylbl, fontdict = font_labels)
    plt.legend()
    # plt.text(0.25, 0.45, artext , name = 'Times New Roman',fontsize=50)
    plt.savefig(address_save)
    plt.show()

def plot_final(x, y, xlabel, ylabel ):
    plt.xticks(rotation=45)
    xlbl = get_display( arabic_reshaper.reshape(xlabel))
    ylbl = get_display( arabic_reshaper.reshape(ylabel))
    x_true = []
    for item in x:
        x_true.append(get_display(arabic_reshaper.reshape(item)))
    plt.bar(x_true,y)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.savefig("../final_F1")
    plt.show()


if __name__ == "__main__":

    # train_and_serialize_models()

    main_proc()
