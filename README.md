DATA 622 # hw2

	Assigned on September 27, 2018
	Due on October 17, 2018 11:59 PM EST
	15 points possible, worth 15% of your final grade

1. Required Reading

	Read Chapter 5 of the Deep Learning Book
	Read Chapter 1 of the Agile Data Science 2.0 textbook

2. Data Pipeline using Python (13 points total)

	Build a data pipeline in Python that downloads data using the urls given below, trains a random forest model on the training dataset using sklearn and scores the model on the test dataset.

	Scoring Rubric

	The homework will be scored based on code efficiency (hint: use functions, not stream of consciousness coding), code cleaniless, code reproducibility, and critical thinking (hint: commenting lets me know what you are thinking!)
Instructions:

	Submit the following 5 items on github.
	ReadMe.md (see "Critical Thinking")
	requirements.txt
	pull_data.py
	train_model.py
	score_model.py

More details:

requirements.txt (1 point)
This file documents all dependencies needed on top of the existing packages in the Docker Dataquest image from HW1. When called upon using pip install -r requirements.txt , this will install all python packages needed to run the .py files. (hint: use pip freeze to generate the .txt file)

pull_data.py (5 points)
When this is called using python pull_data.py in the command line, this will go to the 2 Kaggle urls provided below, authenticate using your own Kaggle sign on, pull the two datasets, and save as .csv files in the current local directory. The authentication login details (aka secrets) need to be in a hidden folder (hint: use .gitignore). There must be a data check step to ensure the data has been pulled correctly and clear commenting and documentation for each step inside the .py file.
	Training dataset url: https://www.kaggle.com/c/titanic/download/train.csv
	Scoring dataset url: https://www.kaggle.com/c/titanic/download/test.csv

train_model.py (5 points)
When this is called using python train_model.py in the command line, this will take in the training dataset csv, perform the necessary data cleaning and imputation, and fit a classification model to the dependent Y. There must be data check steps and clear commenting for each step inside the .py file. The output for running this file is the random forest model saved as a .pkl file in the local directory. Remember that the thought process and decision for why you chose the final model must be clearly documented in this section.
eda.ipynb (0 points)

[Optional] This supplements the commenting inside train_model.py. This is the place to provide scratch work and plots to convince me why you did certain data imputations and manipulations inside the train_model.py file.

score_model.py (2 points)
When this is called using python score_model.py in the command line, this will ingest the .pkl random forest file and apply the model to the locally saved scoring dataset csv. There must be data check steps and clear commenting for each step inside the .py file. The output for running this file is a csv file with the predicted score, as well as a png or text file output that contains the model accuracy report (e.g. sklearn's classification report or any other way of model evaluation).

3. Critical Thinking (2 points total)
Modify this ReadMe file to answer the following questions directly in place.

	1) Kaggle changes links/ file locations/login process/ file content

The current version of the pull_data.py script could be made more robust by using the Kaggle data API. This would likely simplify the login process and introduce more stability into the application.

The path to files and the URLs are parameters in the current version of the pull_data.py script so they can be changed easily.

The current version of the pull_data.py script is robust to changes in the ordering of the columns, but not to changes in the names of the columns, as this would have an impact downstream in the feature creation and training parts of the application.

If the file content changes, we can modify the pull_data.py script to handle the changes, but this could also require the entire application to be redeveloped. To make the script more robust we can add more checks in the data read function to ensure that the column names and the width of the data are the same. This is a very simple modification in the error checking that could reduce the risk that data is passed to the training and scoring parts of the application in the wrong format. 



	2) We run out of space on HD / local permissions issue - can't save files

Without permissions to write files we might need to be able to do everything in memory. We can pass input and output between functions without writing to disk. The pull_data.py bundle of functions reads the training and testing data directly from the internet into a DataFrame. Writing the data to disk is only used as persistent storage and may not be strictly required. It would not be overly complex to change the scripts so that output is piped between scripts or to consolidate the scripts so the the DataFrame is passed between sections without the need for persistent storage.

Redis - an open source, in-memory data structure store that stores data values of several different types associated with a given key - can be used as a cache for persisting objects beyond the lifespan of a Python process. Redis is available on Docker and thus could be used to pass DataFrames between the three main scripts (i.e., pull_data.py, train_data.py, and score_model.py).

Alternatively, we can move the container to a different server with the necessary resources/permissions, pull the application from the source control repository,  and run the application without changes. Storage can be mounted inside the container so we can generally get around permission issues without needing to move the application to another server.


	3) Someone updated python packages and there is unintended effect (functions retired or act differently)

The docker container and the requirements.txt file allow us to restore the original state of the environment. To use the application  after code-breaking changes are made to package versions we can restore the state of our original application. We can then develop an updated version of the application with updated packages if really necessary without disrupting use of the application.

To make the application robust to such potential changes in package versions we can add code to check the package version and throw an error if package versions do not match the versions specified in the requirements.txt

	4) Docker issues - lost internet within docker due to some ip binding to vm or local routing issues( I guess this falls under lost internet, but I am talking more if docker is the cause rather then ISP)

We can add error checking that checks for network/internet access via ping of common high-availability sites (such as www.google.com). Network checks can be very complex or very simple depending on our application requirements.

