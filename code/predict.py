import os
import time

import numpy as np
import tensorflow as tf

from model import FC_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Path to the npy file')
flags.DEFINE_string('weights_path', '', 'Path to the saved weights')
flags.DEFINE_string('prediction_path', '', 'Path to save prediction npy')
# flags.DEFINE_string('mean_value_path', '', 'Path to npy which saves mean values of training samples')
# flags.DEFINE_string('std_value_path', '', 'Path to npy which saves std values of training samples')

flags.DEFINE_integer('input_shape', 2, 'Same as the parameter defined in train.py')
flags.DEFINE_integer('num_outputs', 2, 'Same as the definition in train.py')
flags.DEFINE_integer('model_type', 0, 'Same as the definition in train.py')
FLAGS = flags.FLAGS

class SFDI_FC_model:
    def __init__(self, params):
        self.model = FC_model(params['input_shape'], params['num_outputs'], 
                              depth = params['depth'], neurons_per_layer = params['neurons_per_layer'], 
                              normalization_flag = params['normalization_flag'], activation_func = params['activation_func'], 
                              model_type = params['model_type'])

        self.model.load_weights(params['weights_path'])

    def predict(self, x_test):

        return self.model.predict(x_test)

def main(_):
    x_test = np.load(FLAGS.input_path)
    
    params = {'input_shape': (FLAGS.input_shape,), 
              'depth': 6, 
              'neurons_per_layer': 8, 
              'num_outputs': FLAGS.num_outputs,
              'normalization_flag': False, 
              'activation_func': 'tanh', 
              'weights_path': FLAGS.weights_path, 
              'model_type': FLAGS.model_type}

    sfdi = SFDI_FC_model(params)
    tic = time.time()
    y_pred = sfdi.predict(x_test)
    toc = time.time()
    print('Time for predicting {:d} samples is {:.3f}s'.format(len(x_test), (toc - tic)))
    y_pred[:, 1] *= 10.0 # Must be compactable with train.py

    np.save(FLAGS.prediction_path, y_pred)

if __name__ == '__main__':
    tf.app.run()          
