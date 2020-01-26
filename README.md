# Wine-Tasting-using-Python-Scikit
Training and tuning a random forest for wine quality using supervised learning

# Requirements:
-Python 2.7+ or Python 3

-NumPy

-Pandas

-Scikit-Learn (a.k.a. sklearn)


Dataset url = http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

# Steps

## 1.Set up your environment

## 2.Import libraries and modules.

## 3.Load red wine data.

## 4.Split data into training and test sets:

Splitting the data into training and test sets at the beginning of your modeling workflow is crucial for getting a realistic estimate of your model's performance.

train_test_split() function used

## 5.Declare data preprocessing steps:

First standardization- Standardization is the process of subtracting the means from each feature and then dividing by the feature standard deviations.

Transformer api for preprocessing- we'll be using a feature in Scikit-Learn called the Transformer API. The Transformer API allows you to "fit" a preprocessing step using the training data the same way you'd fit a model.

Here's what that process looks like:

1.Fit the transformer on the training set (saving the means and standard deviations)

scaler = preprocessing.StandardScaler().fit(X_train)

2.Apply the transformer to the training set (scaling the training data)

X_train_scaled = scaler.transform(X_train)

3.Apply the transformer to the test set (using the same means and standard deviations)

X_test_scaled = scaler.transform(X_test)

## 6.Declare hyperparameters to tune:

Hyperparameters express "higher-level" structural information about the model, and they are typically set before training the model.

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

## 7.Tune model using cross-validation pipeline:

Cross-validation is a process for reliably estimating the performance of a method for building a model by training and evaluating your model multiple times using the same method.

clf = GridSearchCV(pipeline, hyperparameters, cv=10)

## 8.Refit on the entire training set

## 9.Evaluate model pipeline on test data.

## 10.Save model for further use.
