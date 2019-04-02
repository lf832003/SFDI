import os
import pickle
import numpy as np
import tensorflow as tf

from model import FC_model

from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

flags = tf.app.flags
flags.DEFINE_string('npy_folder', '', 'Path to the folder containing training and validation npy files')
flags.DEFINE_string('weights_path', '', 'Path to the folder to store the weights of model')
flags.DEFINE_integer('input_shape', 2, 'Model input shape, 2 for 2 frequency, 3 for 3 frequency etc.')

flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs')
flags.DEFINE_integer('num_outputs', 2, 'Number of outputs') # 2 for mua and musp, 1 for mua or musp
flags.DEFINE_string('pre_trained_weights', '', 'Path to pretrained weights')
flags.DEFINE_integer('model_type', 0, 'Model selection, see model.py')
FLAGS = flags.FLAGS

def train_model(x_train, y_train, x_val, y_val, params):
    sfdi = FC_model(params['input_shape'], params['num_outputs'], 
                    depth = params['depth'], neurons_per_layer = params['neurons_per_layer'], 
                    lamda = params['lamda'], dropout_rate = params['dropout_rate'], 
                    normalization_flag = params['normalization_flag'], activation_func = params['activation_func'], 
                    model_type = params['model_type'])
    
    if params['pre_trained_weights'] != '':
        sfdi.load_weights(params['pre_trained_weights'])

    sfdi.compile(optimizer = Adam(0.001), loss = mean_squared_error)

    loss_train = []
    loss_val = []
    min_loss_v = 1.0
    for i in range(params['num_epochs']):
        print('Epochs {}'.format(i))
        ridx = np.random.choice(len(x_train), len(x_train), replace = False)
        x_train = x_train[ridx]
        y_train = y_train[ridx]

        num_batches = len(x_train) // params['batch_size']
        loss_t = []
        loss_v = []
        for j in range(num_batches):
            x_t = x_train[j * params['batch_size']:(j + 1) * params['batch_size']]
            y_t = y_train[j * params['batch_size']:(j + 1) * params['batch_size']]

            loss_t.append(sfdi.train_on_batch(x_t, y_t))
            if (j + 1) % 10 == 0: # calculate the loss on validation set every 10 mini-batch
                loss_v.append(sfdi.evaluate(x_val, y_val))

        loss_train.append(np.mean(np.asarray(loss_t)))
        loss_val.append(np.mean(np.asarray(loss_v)))
        print('Training loss is {:.6f} and validation loss is {:.6f}.'.format(np.mean(np.asarray(loss_t)), np.mean(np.asarray(loss_v))))

        if loss_val[-1] < min_loss_v:
            min_loss_v = loss_val[-1]
            sfdi.save_weights(os.path.join(params['weights_path'], 'weights_' + str('{:.6f}').format(min_loss_v) + '.h5'))

    with open('Training_loss.pkl', 'w') as fid:
        pickle.dump(loss_train, fid)

    with open('Validation_loss.pkl', 'w') as fid:
        pickle.dump(loss_val, fid)

def main(_):
    x_train = np.load(os.path.join(FLAGS.npy_folder, 'x_train.npy'))
    y_train = np.load(os.path.join(FLAGS.npy_folder, 'y_train.npy'))
    x_val = np.load(os.path.join(FLAGS.npy_folder, 'x_validation.npy'))
    y_val = np.load(os.path.join(FLAGS.npy_folder, 'y_validation.npy'))

    y_train[:, 0] *= 10.0 # could also y_train[:, 1] /= 10.0, mua = y_train[:, 0], musp = y_train[:, 1]
    y_val[:, 0] *= 10.0

    params = {'input_shape': (FLAGS.input_shape,), 
              'depth': 6, 
              'neurons_per_layer': 8, 
              'num_outputs': FLAGS.num_outputs, 
              'lamda': [], 
              'dropout_rate': [], 
              'normalization_flag': False, 
              'activation_func': 'tanh', 
              'num_epochs': FLAGS.num_epochs, 
              'batch_size': FLAGS.batch_size, 
              'weights_path': FLAGS.weights_path,
              'pre_trained_weights': FLAGS.pre_trained_weights, 
              'model_type': FLAGS.model_type}

    train_model(x_train, y_train, x_val, y_val, params)

if __name__ == '__main__':
    tf.app.run()
