# Wine Classification Ensemble Model
Ensemble learning model for wine classification using SVM, MLP, and Logistic Regression in a stacking approach; features model training, prediction, and evaluation with K-fold cross-validation

## Overview
This repository contains code for an ensemble machine learning model designed to classify wine types based on their chemical properties. The model uses a stacking technique combining predictions from a Support Vector Machine (SVM) and a Multi-Layer Perceptron (MLP) classifier, which are then used as inputs to a Logistic Regression model to make the final prediction.

## Dataset
The dataset, `wine.data.csv`, includes various chemical attributes of wines, such as alcohol content, malic acid, ash, and others. Each record in the dataset corresponds to a wine sample, labeled with its classification type.

## Model Workflow
The process is divided into several steps, implemented to train and evaluate the ensemble model effectively:

- **Data Preparation**: The dataset is loaded, labels are separated from the features, and the data is converted to NumPy arrays for processing.
  
- **Initial Training**: An SVM and an MLP classifier are trained on a subset of the training data.
  
- **Prediction Phase**: Both classifiers are used to predict wine types on a validation set. Predictions from both models are then combined to form a new set of features.
  
- **Meta Learning**: A Logistic Regression model is trained on these new features against the validation labels, learning to effectively combine the predictions from the initial models.
  
- **Ensemble Prediction**: For testing, predictions from SVM and MLP are again combined and passed to the trained Logistic Regression model to obtain the final ensemble predictions.
  
- **Evaluation**: The accuracy of each model, including the ensemble model, is evaluated on a separate test set.

## K-Fold Cross-Validation
The model uses 10-fold cross-validation to ensure robustness and generalizability. It splits the data into 10 parts, iteratively using 9 for training and validation and 1 for testing. This process helps in understanding the model's performance across different subsets of data.

## Results
The script calculates and prints the average accuracies of the SVM, MLP, and the Logistic Regression (ensemble model) across all folds. These metrics provide insight into the effectiveness of individual models as well as the combined approach.
