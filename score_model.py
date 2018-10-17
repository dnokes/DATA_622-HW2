#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 09:11:43 2018

@author: dnokes
"""
import pandas
from numpy.core.umath_tests import inner1d
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle

# read model file (pickle), return model object from disk
def readModel(output_directory,model_file_name):
    # open model file handle
    pickleFileHandle = open(output_directory+model_file_name, 'rb')
    # load model
    rf_model = pickle.load(pickleFileHandle)
    # close model file handle
    pickleFileHandle.close()
        
    return rf_model

# read split train/test data from disk
def readModelInputData(output_directory):
    # read full train, and train/test split data
    X_train=pandas.read_csv(output_directory+'X_train.csv')
    X_test=pandas.read_csv(output_directory+'X_test.csv')
    y_train=pandas.read_csv(output_directory+'y_train.csv',squeeze=True)
    y_test=pandas.read_csv(output_directory+'y_test.csv',squeeze=True)
    X=pandas.read_csv(output_directory+'X.csv')
    y=pandas.read_csv(output_directory+'y.csv',squeeze=True)
        
    return X_train, X_test, y_train, y_test,X,y

# read model object and train/test data from disk
def readModelInputs(output_directory,model_file_name):
    try:    
        # read model file (pickle), return model object
        rf_model = readModel(output_directory,model_file_name)
    
    except:
        raise Exception('Could not open model input file: '+str(model_file_name))
        
    try:
        # read model input data
        X_train, X_test, y_train, y_test,X,y=readModelInputData(output_directory)
    except:
        raise Exception('Could not read test and train data.')
    
    return rf_model,X_train, X_test, y_train, y_test,X,y

def saveScoreReport(output_directory,class_report_train,class_report_test,score_train,score_test):
    
    try:
        outputFileHandle=open(output_directory+'score_model_report.txt','w')
        outputFileHandle.write('In-Sample Performance:\n')
        outputFileHandle.write(class_report_train+"\n")
        outputFileHandle.write('In-Sample Accuracy: '+str(round(score_train,4))+'\n')
        outputFileHandle.write('Out-Of-Sample Performance:\n')
        outputFileHandle.write(class_report_test+'\n')
        outputFileHandle.write('Out-Of-Sample Accuracy: '+str(round(score_test,4))+'\n')
        outputFileHandle.close()
    except:
        raise Exception('Did not write scoring report to disk.')
        raise Exception('Please check your write permissions')
        
    return

def scoreModel(output_directory,model_file_name):
    # read model inputs
    rf_model,X_train, X_test, y_train, y_test,X,y=readModelInputs(output_directory,
        model_file_name)
    # compute accuracy for model on training set
    score_train = rf_model.score(X_train, y_train)
    # create training set predictions
    y_pred_train = rf_model.predict(X_train)
    # create classification report for train set
    class_report_train = classification_report(y_train, y_pred_train)
    print('In-Sample Performance:')
    print(class_report_train)
    print('In-Sample Accuracy: '+str(round(score_train,4)))
    # compute accuracy for model on test set
    score_test = rf_model.score(X_test, y_test)
    # create test set predictions
    y_pred = rf_model.predict(X_test)
    # create classification report
    class_report_test = classification_report(y_test, y_pred)
    print('Out-Of-Sample Performance:')
    print(class_report_test)
    print('Out-Of-Sample Accuracy: '+str(round(score_test,4)))
    
    # save scoring report to disk
    saveScoreReport(output_directory,class_report_train,class_report_test,score_train,score_test)    
    
    return y_pred

# pass in model, X, y, and number of folds, return mean accuracy and 2x standard 
# deviation of accuracy based on cross-validation
def crossValidation(rf_model, X, y,nFolds):
    # compute cross validation score for each of nFolds different splits
    scores = cross_val_score(rf_model, X, y, cv=nFolds)
    # compute mean score and the 95% confidence interval of the score estimate
    print('Cross-Validation Results (nFolds='+str(nFolds)+'):')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# we were not explicitly instructed to create model predicitons for the hold-out
# data so this is a place holder for later expansion.
def predictHoldOut():
    
    return

# read model inputs, compute accuracy, precision, recall, and f1-score for both 
# in-sample and out-of-sample. Conduct cross-validation to provide indication of 
# variability of accuracy. EDA notebook also contains the confusion matrix - both
# normalized and non-normalized.
def runModelPerformance(output_directory,model_file_name):
    # read model inputs
    rf_model,X_train, X_test, y_train, y_test,X,y=readModelInputs(output_directory,
        model_file_name)
    # define number of folds for cross-validation
    nFolds=5
    # score model in- and out- of-sample (for comparison)
    y_pred=scoreModel(output_directory,model_file_name)
    # conduct cross-validation to assess variability of performance
    crossValidation(rf_model, X, y,nFolds)

    # write test prediction
    pandas.DataFrame(y_pred,columns=['Survived']).to_csv(output_directory+'y_test_prediction.csv',
        index=False)
    
    return

# define input directory
output_directory=''
# define model output file name
model_file_name='rf_model.pkl'
# score
runModelPerformance(output_directory,model_file_name)