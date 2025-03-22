import pytest
# TODO: add necessary import
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics
from ml.data import process_data


# TODO: implement the first test. Change the function name and input as needed
def test_data_shape_check():
    """
    This tests the data to make sure it is dimensional and is not an empty array.
    Process:
        Load data path.
        Check data dimensionality.
            If size == 0, give message
    """
    dataframe = pd.read_csv(os.path.join(os.getcwd(), "data", "census.csv"))
    
    if dataframe.shape[0]==0:
        print("This file contains no rows.")
    elif dataframe.shape[1]==0:
        print("This file contains no columns.")
    else:
        print("This file has both rows and columns. Carry on!")


# TODO: implement the second test. Change the function name and input as needed
def test_model_validity_check():
    """
    This test is designed to ensure a RandomForestClassifier model is created.
    Process:
        Create arrays to pass to called functions.
        Create model.
        Check to ensure model is a RandomForestClassifier.
    """
    test_x = np.random.rand(10,5)
    test_y = np.random.randint(0,2,10)
    
    test_model = train_model(test_x, test_y)

    if isinstance(test_model,RandomForestClassifier):
        print ("This is a RandomForestClassifier model. You may proceed!")
    else:
        print("This is NOT a RandomForestClassifier model. Please try again.")


# TODO: implement the third test. Change the function name and input as needed
def test_model_metrics_check():
    """
    This test is designed to run the metrics of a model and ensure they are in range.
    Process:
        Creates two random arrays to give to .
        Run compute_model_metrics to get Precision, Recall, and fbeta.
        Check these values fall between 0 and 1
    """
    test_predictions = np.random.randint(0,2,5)
    test_y = np.random.randint(0,2,5)

    precision, recall, fbeta = compute_model_metrics(test_y,test_predictions)

    # I learned about the assert function while googling and will be using it for
    # this test.
    assert (0<=precision<=1), "Precision is not between 0 and 1."
    assert (0<=recall<=1), "Recall is not between 0 and 1."
    assert (0<=fbeta<=1), "Fbeta is not between 0 and 1."
