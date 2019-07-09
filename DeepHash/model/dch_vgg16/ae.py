import os
import time
from math import ceil

import numpy as np
import tensorflow as tf

from architecture import img_inception_layers
from architecture import temporal_layers
from evaluation import MAPs
import model.plot as plot

class AutoencoderHashing(object):
    def __init__(self, config):
        print("initializing autoencoder")
        np.set_printoptions(precision=4)

        with.tf.name_scope("stage"):
            self.stage = tf.placeholder_with_default(tf.constant(0), [])
        for k, v in vars(config).items():
            setattr(self, k, v)

        self.file_name = 'aeh_lr_{}_cqlambda_{}_alpha_{}_bias_{}_gamma_{}_dataset_{}'.format(
                self.lr,
                self.q_lambda,
                self.alpha,
                self.bias,
                self.gamma,
                self.dataset)
        self.save_file = os.path.join(self.save_dir, self.file_name + '.npy')

        print("launching session")
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_Placement = True
        self.sess = tf.Session(config=configProto)

        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = self.apply_loss_function(self.global_step)
        self.sess.run(tf.global_variables_initializer())
        return

    def load_model(self):
        if self.img_model == 'inception':
            img_output = img_inception_layers(
                self.img,
                self.batch_size,
                self.output_dim,
                self.stage,
                self.model_weights,
                self.with_tanh,
                self.val_batch_size)
        else:
            raise Exception('model no found ' + self.img_model)
        return img_output

    def save_model(self, model_file=None):
        if model_file is None:
            model_file = self.save_file
        model = {}
        for layer in self.deep_param_img:
            model[layer] = self.sess.run(self.deep_param_img[layer])
        print("saving model to %s" % model_file)
        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)

        np.save(model_file, np.array(model))

    def train(self, train_dataset):
        print("%s #train# start training" % datetime.now())
        
        # tensorboard
        tflog_path = os.path.join(self.log_dir, self_file_name)
        if os.path.exists(tflog_path):
            shutil.rmtree(tflog_path)
        train_writer = tf.summary.FileWriter(tflog_path, self.sess.graph)
        
        # train loop
        for train_iter in range(self.iter_num):
            images, labels = img_dataset.next_batch(self.batch_size)
            start_time = time.time()

            _, loss, cos_loss, output, summary = self.sess.run([self.train_op, self.loss, self.],
                                                               feed_dict={self.img: images}) # pending data format
            train_writer.add_summary(summary, train_iter)

            img_dataset.feed_batch_output(self.batch_size, output)
            duration = time.time() - start_time

            if train_iter % 100 == 0:
                print("%s #train# step %4d, loss = %.4f, cross_entropy loss = %.4f, %.1f sec/batch"
                        %(datetime.now(), train_iter+1, loss, cos_loss, duration))
        print("%s #train# finish training" % datetime.now())
        self.save_model()
        print("model saved")
        self.sess.close()

    def validation(self, img_query, img_database, R=100):
        print("%s #validation# start training" % datetime.now())
        # pending
