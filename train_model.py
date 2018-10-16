#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 09:11:40 2018

@author: dnokes
"""
import numpy
import pandas
from numpy.core.umath_tests import inner1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import pickle

def readData(output_directory,output_file_train):
    try:
        train_df = pandas.read_csv(output_directory+output_file_train)
    except:
        print("Could not read training data.")
        print("Please check to ensure that the path and file name are correct")
    
    return train_df 

def nRooms(x):
    try:
        n=len(x)
    except:
        n=x
    return n

def addCabinFeatures(train_df):
    # remove room numbers and split on space
    train_df['Level']=train_df['Cabin'].str.replace('\d+','').str.split()
    # find number of rooms
    train_df['nRooms']=train_df['Level'].apply(lambda x : nRooms(x))
    # fill missing values with 0
    train_df['nRooms'].fillna(value=0,inplace=True)
    # concatenate the level strings for those with multiple rooms
    train_df['cLevel']=train_df['Level'].str.join('').values
    # fill missing values with 'Missing'
    train_df['cLevel'].fillna(value='Missing',inplace=True)    

    return train_df

def addAgeGroupFeatures(train_df):
    missingValue=-0.5
    bins = [-1,0,5,12,18,35,60,100]
    labelNames = ["Missing","0_5","5_12","12_18","18_35","35_60","60_100"]
    train_df["AgeFilled"] = train_df["Age"].fillna(missingValue)
    train_df["ageGroup"] = pandas.cut(train_df["AgeFilled"],bins,labels=labelNames)
    
    return train_df

# define functions for adding dummy variables
def addDummyVariable(df,columnName):
    df = pandas.concat([df,pandas.get_dummies(df[columnName],prefix=columnName)],axis=1)
    return df

def iterateAddDummyVariables(df,columnNames):
    for columnName in columnNames:
        df = addDummyVariable(df,columnName)    
    return df

def buildFeatures(train_df):
    # Drop the 'Name' and 'Embarked' variables without deriving any features from these fields. 
    # 'Embarked' appeared to be independent of survival rates in the exploratory data analysis. 
    # 'Name' could likely be used to create features indicating the social status of an individual, 
    # but such features are not explored here.
    # Features were derived from 'Age' (i.e., 'ageGroup') and 'Cabin' (i.e., nRooms). 
    # Dummy variables were created from 'Pclass', 'Sex', 'ageGroup', 'Parch', and 'SibSp'.
    # Fare was included without any processing  
    # Please see eda notebook for more details
    try:
        # add features derived from 'Cabin'
        train_df=addCabinFeatures(train_df)
        # add features derived from 'Age'
        train_df=addAgeGroupFeatures(train_df)
        # define columns for conversion to dummy variables
        dummyColumnNames=['Sex','Pclass','ageGroup','SibSp','Parch']
        # create dummy variables for gender, class, age group, embarked code, sibling/spouse,
        df_train=iterateAddDummyVariables(train_df,dummyColumnNames)
        # drop unneeded predictors 
        df_train.drop(['PassengerId','Pclass','Name','Sex','SibSp','Parch','Ticket',
            'Age','AgeFilled','ageGroup','Cabin','Embarked','Level','cLevel','Sex_male'], 
            axis=1,inplace=True)
    except:
        print("Could not build features.")
        print("Please check to ensure that the correct data is being passed into the function")
    
    return df_train

def buildModelPipeline(randomSeed):
    try:
        # define imputation
        rf_impute = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        # define random forest classifier
        rf = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, 
            random_state=randomSeed)
        # build pipeline
        pipeline = Pipeline([('imputation', rf_impute), ('random_forest', rf)])    
    except:
        print("Could not build pipeline")
    
    return pipeline

# take pipeline, training data train/test split parameters as input, train model
# and return trained model and train/test data as output
def trainModel(pipeline,df_train,randomSeed,test_size_percent):
        
    # extract predictors
    X = df_train.drop("Survived",axis=1)
    # extract target
    y = df_train["Survived"]
    
    # oerform train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percent, 
        random_state=randomSeed)
    # fit model using pipeline defined above
    rf_model = pipeline.fit(X_train, y_train)    
        
    return rf_model,X_train, X_test, y_train, y_test,X,y

# save model (pickle)
def saveModel(rf_model,output_directory,model_file_name):

    try:
        # open file handle
        pickleFileHandle = open(output_directory+model_file_name, 'wb')
        # save model to file
        pickle.dump(rf_model, pickleFileHandle)

    except:
        print("Could not save model to file.")
    
    # close file handle
    pickleFileHandle.close()    
    
    return

# save split train/test data from original training data
def saveData(output_directory,X_train, X_test, y_train, y_test,X,y):
    try:
        # save train and test data
        X_train.to_csv(output_directory+'X_train.csv',index=False)
        X_test.to_csv(output_directory+'X_test.csv',index=False)
        y_train.to_csv(output_directory+'y_train.csv',index=False,header="Survived")
        y_test.to_csv(output_directory+'y_test.csv',index=False,header="Survived")
        X.to_csv(output_directory+'X.csv',index=False)
        y.to_csv(output_directory+'y.csv',index=False,header="Survived")
    except:
        print("Could not save test and train data.")
        
    return 

# print feature importance
def printFeatureImportance(pipeline,X):
    # extract feature importance
    importances = pipeline.steps[1][1].feature_importances_
    # extract column names
    columnNames=X.columns.values
    # create index for sort by importance
    indices = numpy.argsort(importances)[::-1]
    print("Feature Importance Ranking:")
    # iterate over each feature and display feature name and importance 
    for f in range(X.shape[1]):
        print("%d. feature: %s (%f)" % (f + 1, columnNames[indices[f]], importances[indices[f]]))
    
    return

# read data, build features, build model pipeline, train model, save model, and
# save split train/test data from original training data set
def runTrainModel():
    # define input directory
    output_directory=''
    # define input training data
    output_file_train='train.csv'
    # define model output file name
    model_file_name='rf_model.pkl'
    # define holdout percent (i.e., set aside 40% of the training set for testing)
    test_size_percent=0.4
    # define random seed
    randomSeed=123456789
    # read data
    train_df=readData(output_directory,output_file_train)
    # build features
    df_train=buildFeatures(train_df)
    # build model pipeline
    pipeline=buildModelPipeline(randomSeed)
    # train model
    rf_model,X_train, X_test, y_train, y_test,X,y=trainModel(pipeline,df_train,randomSeed,
        test_size_percent) 
    # print feature importance
    printFeatureImportance(pipeline,X_train)
    # save model (pickle)
    saveModel(rf_model,output_directory,model_file_name)
    # save test/train output
    saveData(output_directory,X_train, X_test, y_train, y_test,X,y)
    
    return

runTrainModel()