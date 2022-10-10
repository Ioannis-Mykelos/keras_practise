"""
functions file
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, List, Sequence

def load_fashion_mnist_data()->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """
    This function loads the fashion mnist dataset
    Arguments:
    ---------------------------------------------
    -None
    Returns:
    ---------------------------------------------
    -X_train  : the X_train pd.DataFrame
    -y_train  : the y_train pd.DataFrame
    -X_test   : the X_test pd.DataFrame
    -y_test   : the y_train pd.DataFrame
    -X_valid  : the X_valid pd.DataFrame
    -y_valid  : the y_valid pd.DataFrame
    """
    # Load the fashion mnist dataset
    print("Loading data ... ")
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # Create train, test & validation sets and scale them
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    # Print shapes and dtypes
    print("X_train shape",X_train.shape)
    print("X_train dtype",X_train.dtype)
    print("X_test shape",X_test.shape)
    print("X_test dtype",X_test.dtype)
    print("X_valid shape",X_valid.shape)
    print("X_valid dtype",X_valid.dtype)
    print("y_train shape",y_train.shape)
    print("y_train dtype",y_train.dtype)
    print("y_test shape",y_test.shape)
    print("y_test dtype",y_test.dtype)
    print("y_valid shape",y_valid.shape)
    print("y_valid dtype",y_valid.dtype)
    
    return X_train, X_test, X_valid, y_train, y_test, y_valid