from keras import backend as K
from keras.layers import Layer

class norm_layer(Layer): # output_dim must be equal to input_shape[1]
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(norm_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mean = self.add_weight(name = 'mean', 
                                    shape = (input_shape[1],),
                                    initializer = 'glorot_uniform', 
                                    trainable = True)
        self.std = self.add_weight(name = 'std', 
                                   shape = (input_shape[1],),
                                   initializer = 'glorot_uniform', 
                                   trainable = True)
        self.bias = self.add_weight(name = 'bias', 
                                    shape = (input_shape[1],), 
                                    initializer = 'glorot_uniform', 
                                    trainable = True)
        super(norm_layer, self).build(input_shape)

    def call(self, x):
        return (x - self.mean) / self.std + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
