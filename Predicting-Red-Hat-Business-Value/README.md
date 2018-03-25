#Predicting Red Hat Business Value

Team name - minions

Team members: Chirag Jain, Sharmin Pathan

Approach: To classify customer potential using a suitable prediction model.

Technologies Used:
-----------------
- Python 2.7
- RFE
- Logistic Regression
- Random Forest Classifier

Introduction:
------------
Like most companies, Red Hat is able to gather a great deal of information over time about the behavior of individuals who interact with them. They’re in search of better methods of using this behavioral data to predict which individuals they should approach—and even when and how to approach them.
With an improved prediction model in place, Red Hat will be able to more efficiently prioritize resources to generate more business and better serve their customers.
(This competition was hosted on Kaggle)

Problem Statement:
-----------------
To create a classification algorithm that accurately identifies which customers have the most potential business value for Red Hat based on their characteristics and activities.

Datatset:
--------
The dataset was taken from https://www.kaggle.com/c/predicting-red-hat-business-value/data.
It includes people.csv, act_train.csv and act_test.csv.
The people file contains all of the unique people (and the corresponding characteristics) that have performed activities over time. Each row in the people file represents a unique person. Each person has a unique people_id.

The activity files contain all of the unique activities (and the corresponding activity characteristics) that each person has performed over time. Each row in the activity file represents a unique activity performed by a person on a certain date. Each activity has a unique activity_id.

Preprocessing:
-------------
- Convert the categorical values into numerical ones
- Convert boolean attributes to numerical ones
- Break date column into three separate columns, namely date, month, and year
- Fill '-1' for the missing values
- Merge people and activity files
- Separate the data and labels

Flow:
----
- Load the datasets into dataframes
- Perform the preprocessing
- Drop date, people_id, and act_id columns
- Perform Logistic Regression for RFE. RFE uses the model accuracy to identify which attributes contribute the most to  
  predicting the target attribute.
- Use a 4-fold cross validation to have 3-folds as the training set and 1-fold for the validation set
- Apply Random Forest classification
- Predict outcomes for activities in the testing data set
- Build a submission file with activities and their corresponding predicted outcomes

Execution:
---------
Ensure the system is up with
- RFE
- Matplotlib

The program takes three command line arguments:
- path to people.csv
- path to act_train.csv
- path to act_test.csv

For example: redHat.py people.csv act_train.csv act_test.csv

Performance:
-----------
We made a submission on Kaggle.

![screen shot 2016-12-13 at 7 14 34 pm](https://cloud.githubusercontent.com/assets/20985174/21169792/5e309dba-c18d-11e6-87fe-e5f78d084322.png)


Tuning the accuracy:
-------------------
- Tuning parameters of Random Forest classifier, namely maxDepth, no. of trees, maxBin.

Stuff we tried:
--------------
- Plotting graphs for various attributes against the outcome to understand their distribution in the dataset
- PCA Visualization

Challenges:
----------
The dataset was pretty difficult to understand. Enough details were not provided on the competition page about the datasets. 

####References: https://www.kaggle.com/c/predicting-red-hat-business-value
