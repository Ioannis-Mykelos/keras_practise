"""
The models file
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    -model_we_have : model we built
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

def fit_model(
    model_we_have:Sequence, 
    x_train_df:pd.DataFrame,  
    y_train_df:pd.DataFrame,
    x_valid_df:pd.DataFrame,  
    y_valid_df:pd.DataFrame,
    )->Sequence:
    """
    This function traind the  sequential model
    Arguments:
    ---------------------------------------------
    -model_we_have : model we built
    Returns:
    ---------------------------------------------
    -model       : a Sequential keras model
    """
    history = model_we_have.fit(x_train_df, y_train_df, epochs=30, validation_data=(x_valid_df, y_valid_df))
    print("Model parameters -->\n", history.params)
    print("")
    print("Model list of epochs it went through -->\n", history.epoch)
    print("")
    print("Models loss and extra metrics it measured at the end of each epoch -->\n", history.history)
    print("")
    print("Visualization")
    #figure(figsize=(20, 15), dpi=150)
    pd.DataFrame(history.history).plot(figsize=(20, 15))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    plt.show()
    return history
