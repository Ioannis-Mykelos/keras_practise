"""
The models file
"""
from msilib import sequence
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, List, Sequence

def create_sequential_model(input_shape_dim:int)->Sequence:
    """
    This function creates a simple sequential model
    Arguments:
    ---------------------------------------------
    -input_shape_dim : the integer that defines the dimension of the picture
    Returns:
    ---------------------------------------------
    -model       : a Sequential keras model
    """
    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[input_shape_dim, input_shape_dim]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])
    # Models summary
    print(model.summary())
    print("")
    print("Layers:")
    for layer in model.layers:
        print(layer)
    return model

def compile_model(model_we_have:Sequence)->Sequence:
    """
    This function compliles sequential model
    Arguments:
    ---------------------------------------------
    -input_shape_dim : the integer that defines the dimension of the picture
    Returns:
    ---------------------------------------------
    -model       : a Sequential keras model
    """
    #Compile the model to specify the loss function and the optimizer to use.
    model_we_have.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"]
    )
    return model_we_have