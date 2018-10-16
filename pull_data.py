# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import getpass
import pandas
import requests
import csv

def getLogin():
    # prompt via command line for username
    username = getpass.getpass("Please enter your Kaggle username:")
    # prompt via command line for password
    password = getpass.getpass("Please enter your Kaggle password:")
    # return username and password
    return ([username,password])

def writeLogin(output_directory,output_filename,username,password):
    # open login output file
    outputFileHandle=open(output_directory+output_filename,'w')
    # write header
    outputFileHandle.write("username|password\n")
    # write credentials
    outputFileHandle.write(str(username)+"|"+str(password)+"\n")
    # close output file handle
    outputFileHandle.close()
    
    return
    
def readLogin(output_directory,output_filename):
    user_info=pandas.read_csv(output_directory+output_filename,sep='|')
    
    return user_info.to_dict(orient='records')[0]

def buildPayload(payload):
    payload['__RequestVerificationToken']=''
    payload['rememberme']='false'
    
    return payload

def fetchKaggleData(login_url,data_url,payload):
    # open session
    with requests.Session() as s:
        # fetch response from login URL
        response = s.get(login_url).text
        # extract anti-forgery token
        AFToken = response[response.index('antiForgeryToken')+19:response.index('isAnonymous: ')-12]
        # add anti-forgery token to credentials
        payload['__RequestVerificationToken']=AFToken
        # login
        s.post(login_url + "?isModal=true&returnUrl=/", data=payload)
        # fetch data
        download = s.get(data_url)
        # decode content
        decoded_content = download.content.decode('utf-8')
        # read csv and convert to list
        data_list = list(csv.reader(decoded_content.splitlines(), delimiter=','))
    # convert data list to dataframe
    df = pandas.DataFrame(data_list)
    # extract header
    header = df.iloc[0]
    # extract data
    df = df[1:]
    # use header to add column names
    kaggle_dataset = df.set_axis(header, axis='columns', inplace=False)
        
    return kaggle_dataset

def fetchTitanicKaggleData(output_directory,login_filename,output_file_train,output_file_test,data_train_url,data_test_url,login_url):
    # try to fetch login and password from user
    try:
        # prompt for Kaggle username and password
        username,password=getLogin()
        # write Kaggle username and password
        writeLogin(output_directory,login_filename,username,password)
        # read Kaggle username and password
        user_info=readLogin(output_directory,login_filename)
        # build payload with Kaggle user credentials
        payload=buildPayload(user_info)
    
        try:
            # fetch Kaggle data (train)
            trainDf=fetchKaggleData(login_url,data_train_url,payload)
            
            if (trainDf.values[0][0]!='<html>'): 

                try:
                    # fetch Kaggle data (test)
                    trainDf.to_csv(output_directory+output_file_train,index=False)
                except:
                    print('Could not write Kaggle train data to disk.')
                    print('Please check that you have write permissions.')               
                
            else:
                print('Could not fetch Kaggle train data.')
                print('Please check Kaggle user name and password.')
                print('Please check the Kaggle login and data url.')
                print('Please ensure that you are connected to the internet.')                
            
        except:
            print('Did not write Kaggle train data to disk.')
         
        try:
            # fetch Kaggle data (train)
            testDf=fetchKaggleData(login_url,data_test_url,payload)
            
            if (trainDf.values[0][0]!='<html>'): 

                try:
                    # fetch Kaggle data (test)
                    testDf.to_csv(output_directory+output_file_test,index=False)
                except:
                    print('Could not write Kaggle train data to disk.')
                    print('Please check that you have write permissions.')               
                
            else:
                print('Could not fetch Kaggle train data.')
                print('Please check Kaggle user name and password.')
                print('Please check the Kaggle login and data url.')
                print('Please ensure that you are connected to the internet.')                
            
        except:
            print('Did not write Kaggle train data to disk.')            

    # exit with message if cannot 
    except:
        print('Please check that you have write permissions')
            
    return

# define parameters
output_directory=''
login_filename='login'
output_file_train='train.csv'
output_file_test='test.csv'
login_url = 'https://www.kaggle.com/account/login'
data_train_url = 'https://www.kaggle.com/c/titanic/download/train.csv'
data_test_url = 'https://www.kaggle.com/c/titanic/download/test.csv'
# fetch Titanic Kaggle data 
fetchTitanicKaggleData(output_directory,login_filename,output_file_train,output_file_test,
    data_train_url,data_test_url,login_url)
