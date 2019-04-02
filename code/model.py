import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Activation, BatchNormalization, Input, Dense, Dropout
from keras.regularizers import l2
from keras import backend as K

from custom_layer import norm_layer

def dense_block(input, num_neurons, lamda, dropout_rate, normalization_flag, activation_func):
    if lamda != []:
        dense_layer = Dense(num_neurons, kernel_regularizer = l2(lamda), bias_regularizer = l2(lamda))(input)
    else:
        dense_layer = Dense(num_neurons)(input)

    if normalization_flag:
        batch_layer = BatchNormalization()(dense_layer)
    else:
        batch_layer = dense_layer

    act_layer = Activation(activation_func)(batch_layer)

    if dropout_rate != []:
        output = Dropout(dropout_rate)(act_layer)
    else:
        output = act_layer

    return output

def FC_model(input_shape, num_outputs = 2, depth = 6, neurons_per_layer = 10, 
             lamda = [], dropout_rate = [], 
             normalization_flag = False, activation_func = 'tanh', model_type = 0): # lamda = 0.0001, dropout_rate = 0.5
    if not isinstance(neurons_per_layer, list):
        neurons_per_layer = [neurons_per_layer] * depth
    else:
        if len(neurons_per_layer) != depth:
            raise ValueError('The length of nuerons_per_layer must be equal to neural nets depth!')

    inputs = Input(input_shape)
    norm = norm_layer(input_shape[0])(inputs)

    if model_type == 0: # Follow the model structure in the paper.       
        for i in range(depth):        
            if i == 0:
                hidden_layer = dense_block(norm, neurons_per_layer[i], 
                                           lamda, dropout_rate, normalization_flag, activation_func) 
            else:
                hidden_layer = dense_block(hidden_layer, neurons_per_layer[i], 
                                           lamda, dropout_rate, normalization_flag, activation_func) 
    elif model_type == 1:
        for i in range(depth): # Not used        
            if i == 0:
                hidden_layer = dense_block(norm, neurons_per_layer[i], 
                                           lamda, dropout_rate, normalization_flag, 'relu') 
            else:
                if i < depth // 2:
                    hidden_layer = dense_block(hidden_layer, neurons_per_layer[i], 
                                               lamda, dropout_rate, normalization_flag, 'relu') 
                else:
                    hidden_layer = dense_block(hidden_layer, neurons_per_layer[i], 
                                               lamda, dropout_rate, normalization_flag, 'tanh')
    elif model_type == 2: # Deep model, use ReLU function as activation function, reducing gradient vanishing 
        for i in range(depth):        
            if i == 0:
                hidden_layer = dense_block(norm, neurons_per_layer[i], 
                                           lamda, dropout_rate, normalization_flag, 'tanh') 
            else:
                if i < 3:
                    hidden_layer = dense_block(hidden_layer, neurons_per_layer[i], 
                                               lamda, dropout_rate, normalization_flag, 'tanh')
                elif i >= 3 and i < depth - 3:
                    hidden_layer = dense_block(hidden_layer, neurons_per_layer[i], 
                                               lamda, dropout_rate, normalization_flag, 'relu')
                else:
                    hidden_layer = dense_block(hidden_layer, neurons_per_layer[i], 
                                               lamda, dropout_rate, normalization_flag, 'tanh')

    dense = Dense(num_outputs)(hidden_layer)
    output = norm_layer(num_outputs)(dense)

    return Model(inputs, output)
