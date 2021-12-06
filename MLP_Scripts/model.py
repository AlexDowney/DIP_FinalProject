import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input


def build_mlp_model():
    INPUT_DIM = 40
    HIDDEN_DIM0 = 80
    HIDDEN_DIM1 = 120
    HIDDEN_DIM2 = 160
    OUTPUT_DIM = 200
    
    # define the architecture of the network
    model = Sequential()
    model.add(
        Input(
                shape=(INPUT_DIM,)
        )
    )
    """model.add(
        Dense(
                HIDDEN_DIM0,
                activation="relu"
        )
    )"""
    model.add(
        Dense(
                HIDDEN_DIM1,
                activation="relu"
        )
    )
    """model.add(
        Dense(
                HIDDEN_DIM2,
                activation="relu"
        )
    )"""
    model.add(
        Dense(
                OUTPUT_DIM
        )
    )
    
    return model


if __name__ == "__main__":
    model = build_mlp_model();
    model.summary()
    print(model.output_shape)

