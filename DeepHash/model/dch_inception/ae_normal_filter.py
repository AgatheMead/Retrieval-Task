import tensorflow as tf
import numpy as np

import os

class Encoder(object):
    def __init__(self, config):
    ### Initialize setting
    print ("initializing encoder")
    np.set_printoptions(precision=4)
    
    with tf.name_scope('stage'):
        self.stage = tf.placeholder_with_default(tf.constant(0), [])
    for k, v in vars(config).item():
        setattr(self, k ,v)
    self.file_name = 'lr_{}_cqlambda_{}_alpha_{}_bias_{}_gamma_{}_dataset_{}'.format(
                self.lr,
                self.q_lambda,
                self.alpha,
                self.bias,
                self.gamma,
                self.dataset)
    self.save_file = os.path.join(self.save_dir, self.file_name + '.npy')

    ### Setup session
    print ("launching session")
    configProto = tf.ConfigProto()
    configProto.gpu_options.allow_growth = True
    configProto.allow_soft_placement = True
    self.sess = tf.Session(config=configProto)

    ### Create variables and placeholders
    self.img = tf.placeholder(tf.float32, [None, 256, 256, 3])
    self.img_label = tf.placeholder(tf.float32, [None, self.label_dim])
    self.img_last_layer, self.deep_param_img, self.train_layers, self.train_last_layer = self.load_model()

    self.global_step = tf.Variable(0, trainable=False)
    self.train_op = self.apply_loss_function(self.global_step)
    self.sess.run(tf.global_variables_initializer())
    return

    def load_model(self):
        if self.ae_model == 'unet':
            # pending as we don't know the function return yet
        else:
            raise Exception('cannot use such model as ' + self.ae_model)
        return ae_model

    def save_model(self):
        if model_file is None:
            model_file = self.save_file
        model = {}
        for layer in self.deep_param_ae:
            model[layer] = self.sess.run(self.deep_param_ae[layer])
        print ("saving ae model to %s" % model_file)
        if os.path.exists(self.save_dir) is None:
            os.makedirs(self.save_dir)

        np.save(model_file, np.array(model))
        return

    
    def apply_loss_function():
        
