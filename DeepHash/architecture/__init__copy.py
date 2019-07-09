import os
import tensorflow as tf
import numpy as np


def img_alexnet_layers(img, batch_size, output_dim, stage, model_weights, with_tanh=True, val_batch_size=32):
    deep_param_img = {}
    train_layers = []
    train_last_layer = []
    print("loading img model from %s" % model_weights)
    net_data = dict(np.load(model_weights, encoding='bytes').item())
    print(list(net_data.keys()))

    # swap(2,1,0), bgr -> rgb
    reshaped_image = tf.cast(img, tf.float32)[:, :, :, ::-1]

    height = 227
    width = 227

    # Randomly crop a [height, width] section of each image
    with tf.name_scope('preprocess'):
        def train_fn():
            return tf.stack([tf.random_crop(tf.image.random_flip_left_right(each), [height, width, 3])
                             for each in tf.unstack(reshaped_image, batch_size)])

        def val_fn():
            unstacked = tf.unstack(reshaped_image, val_batch_size)

            def crop(img, x, y): return tf.image.crop_to_bounding_box(
                img, x, y, width, height)

            def distort(f, x, y): return tf.stack(
                [crop(f(each), x, y) for each in unstacked])

            def distort_raw(x, y): return distort(lambda x: x, x, y)

            def distort_fliped(x, y): return distort(
                tf.image.flip_left_right, x, y)
            distorted = tf.concat([distort_fliped(0, 0), distort_fliped(28, 0),
                                   distort_fliped(
                                       0, 28), distort_fliped(28, 28),
                                   distort_fliped(14, 14), distort_raw(0, 0),
                                   distort_raw(28, 0), distort_raw(0, 28),
                                   distort_raw(28, 28), distort_raw(14, 14)], 0)

            return distorted
        distorted = tf.cond(stage > 0, val_fn, train_fn)

        # Zero-mean input
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[
                           1, 1, 1, 3], name='img-mean')
        distorted = distorted - mean

    # Conv1
    # Output 96, kernel 11, stride 4
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(net_data['conv1'][0], name='weights')
        conv = tf.nn.conv2d(distorted, kernel, [1, 4, 4, 1], padding='VALID')
        biases = tf.Variable(net_data['conv1'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv1'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    # LRN1
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(pool1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # Conv2
    # Output 256, pad 2, kernel 5, group 2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(net_data['conv2'][0], name='weights')
        group = 2

        def convolve(i, k): return tf.nn.conv2d(
            i, k, [1, 1, 1, 1], padding='SAME')
        input_groups = tf.split(lrn1, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k)
                         for i, k in zip(input_groups, kernel_groups)]
        # Concatenate the groups
        conv = tf.concat(output_groups, 3)

        biases = tf.Variable(net_data['conv2'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv2'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Pool2
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')

    # LRN2
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(pool2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # Conv3
    # Output 384, pad 1, kernel 3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(net_data['conv3'][0], name='weights')
        conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Conv4
    # Output 384, pad 1, kernel 3, group 2
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(net_data['conv4'][0], name='weights')
        group = 2

        def convolve(i, k): return tf.nn.conv2d(
            i, k, [1, 1, 1, 1], padding='SAME')
        input_groups = tf.split(conv3, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k)
                         for i, k in zip(input_groups, kernel_groups)]
        # Concatenate the groups
        conv = tf.concat(output_groups, 3)
        biases = tf.Variable(net_data['conv4'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Conv5
    # Output 256, pad 1, kernel 3, group 2
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(net_data['conv5'][0], name='weights')
        group = 2

        def convolve(i, k): return tf.nn.conv2d(
            i, k, [1, 1, 1, 1], padding='SAME')
        input_groups = tf.split(conv4, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k)
                         for i, k in zip(input_groups, kernel_groups)]
        # Concatenate the groups
        conv = tf.concat(output_groups, 3)
        biases = tf.Variable(net_data['conv5'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')

    # FC6
    # Output 4096
    with tf.name_scope('fc6'):
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc6w = tf.Variable(net_data['fc6'][0], name='weights')
        fc6b = tf.Variable(net_data['fc6'][1], name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6 = tf.nn.relu(fc6l)
        fc6 = tf.cond(stage > 0, lambda: fc6, lambda: tf.nn.dropout(fc6, 0.5))
        fc6o = tf.nn.relu(fc6l)
        deep_param_img['fc6'] = [fc6w, fc6b]
        train_layers += [fc6w, fc6b]

    # FC7
    # Output 4096
    with tf.name_scope('fc7'):
        fc7w = tf.Variable(net_data['fc7'][0], name='weights')
        fc7b = tf.Variable(net_data['fc7'][1], name='biases')
        fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7l)
        fc7 = tf.cond(stage > 0, lambda: fc7, lambda: tf.nn.dropout(fc7, 0.5))
        deep_param_img['fc7'] = [fc7w, fc7b]
        train_layers += [fc7w, fc7b]

    # FC8
    # Output output_dim
    with tf.name_scope('fc8'):
        # Differ train and val stage by 'fc8' as key
        if 'fc8' in net_data:
            fc8w = tf.Variable(net_data['fc8'][0], name='weights')
            fc8b = tf.Variable(net_data['fc8'][1], name='biases')
        else:
            fc8w = tf.Variable(tf.random_normal([4096, output_dim],
                                                dtype=tf.float32,
                                                stddev=1e-2), name='weights')
            fc8b = tf.Variable(tf.constant(0.0, shape=[output_dim],
                                           dtype=tf.float32), name='biases')
        fc8l = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)
        if with_tanh:
            fc8_t = tf.nn.tanh(fc8l)
        else:
            fc8_t = fc8l

        def val_fn1():
            concated = tf.concat([tf.expand_dims(i, 0)
                                  for i in tf.split(fc8_t, 10, 0)], 0)
            return tf.reduce_mean(concated, 0)
        fc8 = tf.cond(stage > 0, val_fn1, lambda: fc8_t)
        deep_param_img['fc8'] = [fc8w, fc8b]
        train_last_layer += [fc8w, fc8b]

    print("img model loading finished")
    # Return outputs
    return fc8, deep_param_img, train_layers, train_last_layer

def temporal_segment_layers(img, batch_size, output_dim, stage, model_weights, with_tanh=True, val_batch_size=32):
    # Inception+reduce_dim model architecture
    # ref: https://github.com/Natsu6767/Inception-Module-Tensorflow
    deep_param_img = {}
    train_layers = []
    train_last_layer = []
    modality = 'Flow' # 'RBGDiff'
    print("loading image model from %s" % model_weights)
    net_data = np.load(model_weights, encoding='bytes')
    #print(list(net_data.keys()))

    # reshape bgr image to rgb image
    reshaped_image = tf.cast(img, tf.float32)[:, :, :, ::-1]

    crop_h, crop_w = (227, 227) # crop to inception default input image size
    # well resize image
    #img = tf.image.resize_images(img, [crop_h, crop_w], method=1)
    
    # Randomly crop a [height, width] section of each image
    with tf.name_scope('preprocess'):
        def train_fn():
            return tf.stack([tf.random_crop(tf.image.random_flip_left_right(each), [crop_h, crop_w, 3])
                             for each in tf.unstack(reshaped_image, batch_size)])

        def val_fn():
            unstacked = tf.unstack(reshaped_image, val_batch_size)

            def crop(img, x, y): return tf.image.crop_to_bounding_box(
                img, x, y, crop_h, crop_w)

            def distort(f, x, y): return tf.stack(
                [crop(f(each), x, y) for each in unstacked])

            def distort_raw(x, y): return distort(lambda x: x, x, y)

            def distort_fliped(x, y): return distort(
                tf.image.flip_left_right, x, y)
            distorted = tf.concat([distort_fliped(0, 0), distort_fliped(28, 0),
                                   distort_fliped(
                                       0, 28), distort_fliped(28, 28),
                                   distort_fliped(14, 14), distort_raw(0, 0),
                                   distort_raw(28, 0), distort_raw(0, 28),
                                   distort_raw(28, 28), distort_raw(14, 14)], 0)

            return distorted
        distorted = tf.cond(stage > 0, val_fn, train_fn) # similar to if-else but tf.cond is evaluated at the runtime while if-else is evaluated at the graph construction time

        # Zero-mean input
        mean = tf.constant([104, 117, 128], dtype=tf.float32, shape=[
                           1, 1, 1, 3], name='img-mean')
        distorted = distorted - mean

    # Conv1: Output 64, pad 1, kernel 3, stride 1
    num_filters=64
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE) as scope:
        #kernel = tf.Variable(net_data['conv1_W'], name='weights') # cannot be created with unknown size
        kernel = tf.get_variable('weights', shape=[3, 3, 3, num_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('biases', shape=[num_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(distorted, kernel, [1, 5, 5, 1], padding='SAME')
        #biases = tf.Variable(net_data['conv1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name='conv1')
        deep_param_img['conv1'] = [kernel, biases]
        train_layers += [kernel, biases]

    pool1 = tf.nn.max_pool2d(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')
    #print('lalala', pool1.get_shape()) # [?,23,23,64]
    inception2a, deep_param_img, train_inception_layers = inception_layers(pool1,
                                                                           deep_param_img,
                                                                           64, 96, 128, 16, 32, 32,
                                                                           name='inception2a')
    train_layers += train_inception_layers
    inception2b, deep_param_img, train_inception_layers = inception_layers(inception2a,
                                                                           deep_param_img,
                                                                           128, 128, 192, 32, 96, 64,
                                                                           name='inception2b')
    train_layers += train_inception_layers
    
    pool2 = tf.nn.max_pool2d(inception2b,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    inception3a, deep_param_img, train_inception_layers = inception_layers(pool2,
                                                                           deep_param_img,
                                                                           192, 96, 208, 16, 48, 64,
                                                                           name='inception3a')
    train_layers += train_inception_layers
    inception3b, deep_param_img, train_inception_layers = inception_layers(inception3a,
                                                                           deep_param_img,
                                                                           160, 112, 224, 24, 64, 64,
                                                                           name='inception3b')
    train_layers += train_inception_layers
    #print(inception3b.get_shape()) (?, 12, 12, 512)
    pool3 = tf.nn.max_pool2d(inception3b,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool3')
    
    inception4a, deep_param_img, train_inception_layers = inception_layers(pool3,
                                                                           deep_param_img,
                                                                           256, 160, 320, 32, 128, 128,
                                                                           name='inception4a')
    train_layers += train_inception_layers
    inception4b, deep_param_img, train_inception_layers = inception_layers(inception4a,
                                                                           deep_param_img,
                                                                           384, 192, 384, 48, 128, 128,
                                                                           name='inception4b')
    train_layers += train_inception_layers
    #print(inception4b.get_shape()) # (?, 6, 6, 1024)
    avg_pool = tf.nn.avg_pool2d(inception4b,
                              ksize=[1, 6, 6, 1],
                              strides=[1, 1, 1, 1],
                              padding='SAME', # used to be 'VALID'
                              name='avg_pool')
    #print(avg_pool.get_shape()) # (?, 1, 1, 1024)
    dropout = tf.nn.dropout(avg_pool, rate=0.5) # Please use `rate` instead of `keep_prob`
    #flatten = tf.contrib.layers.flatten(dropout) # before FC

    # fc5 - Output (9216, 4096)
    with tf.variable_scope('fc5', reuse=tf.AUTO_REUSE):
        shape = int(np.prod(dropout.get_shape()[1:]))
        fc5w = tf.get_variable('weights', shape=[shape, 4096],
                               initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc5b = tf.get_variable('biases', shape=[4096], initializer=tf.constant_initializer(0.0))
        pool5_flat = tf.reshape(dropout, [-1, shape])
        fc5l = tf.nn.bias_add(tf.matmul(pool5_flat, fc5w), fc5b)
        fc5 = tf.nn.relu(fc5l)
        fc5 = tf.cond(stage > 0, lambda: fc5, lambda: tf.nn.dropout(fc5, 0.5))
        fc5o = tf.nn.relu(fc5l)
        deep_param_img['fc5'] = [fc5w, fc5b]
        train_layers += [fc5w, fc5b]

    # fc6 - Output (4096, 4096)
    with tf.variable_scope('fc6', reuse=tf.AUTO_REUSE):
        fc6w = tf.get_variable('weights', shape=[4096, 4096],
                               initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc6b = tf.get_variable('biases', shape=[4096], initializer=tf.constant_initializer(0.0))
        fc6l = tf.nn.bias_add(tf.matmul(fc5, fc6w), fc6b)
        fc6 = tf.nn.relu(fc6l)
        fc6 = tf.cond(stage > 0, lambda: fc6, lambda: tf.nn.dropout(fc6, 0.5))
        deep_param_img['fc6'] = [fc6w, fc6b]
        train_layers += [fc6w, fc6b]
    
    # fc7 layer - Ouput (None, 64)
    with tf.variable_scope('fc7', reuse=tf.AUTO_REUSE):
        fc7w = tf.get_variable('weights', shape=[4096, output_dim],
                               initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc7b = tf.get_variable('biases', shape=[output_dim], initializer=tf.constant_initializer(0.0))
        #pool_flat = tf.reshape(dropout, [-1, 4096])
        fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        if with_tanh:
            fc7_t = tf.nn.tanh(fc7l)
        else:
            fc7_t = fc7l
        def val_fn1():
            concated = tf.concat([tf.expand_dims(i, 0) for i in tf.split(fc7_t, 10, 0)], 0)
            return tf.reduce_mean(concated, 0)
        fc7 = tf.cond(stage > 0, val_fn1, lambda: fc7_t)
        deep_param_img['fc7'] = [fc7w, fc7b]
        train_last_layer += [fc7w, fc7b]
        #print('++', fc7.get_shape()) # ('++', TensorShape([Dimension(None), Dimension(64)]))

    return fc7, deep_param_img, train_layers, train_last_layer # currently tsn only has 1 fc layer
    
def inception_layers(x, deep_param_img, conv1_filters, conv3_r_filters, conv3_filters, conv5_r_filters, conv5_filters, pool_proj_filters, name='inception'):
    # frame_stride=4 # RGB_diff: optical_flow
    # ref: https://github.com/Natsu6767/Inception-Module-Tensorflow
    train_inception_layers = []
    ''' create an Inception layer '''
    # Conv1: Output 64, pad 1, kernel 3, stride 1
    input_channels = int(x.get_shape()[-1])
    name_conv1 = '{}_1x1'.format(name)
    with tf.variable_scope(name_conv1, reuse=tf.AUTO_REUSE) as scope:
        #kernel = tf.Variable(net_data['conv1_W'], name='weights')
        kernel = tf.get_variable('weights', shape=[1, 1, input_channels, conv1_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('biases', shape=[conv1_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        #biases = tf.Variable(net_data['conv1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name='{}_1x1'.format(name))
        deep_param_img['{}_1x1'.format(name)] = [kernel, biases]
        #train_layers += [kernel, biases]
    
    # Conv3_reduce: Output 64, pad 1, kernel 3, stride 1
    input_channels = int(x.get_shape()[-1])
    name_conv3r = '{}_3x3_reduce'.format(name)
    with tf.variable_scope(name_conv3r, reuse=tf.AUTO_REUSE) as scope:
        #kernel = tf.Variable(net_data['conv1_W'], name='weights')
        kernel = tf.get_variable('weights', shape=[1, 1, input_channels, conv3_r_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('biases', shape=[conv3_r_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        #biases = tf.Variable(net_data['conv1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_r = tf.nn.relu(out, name='{}_3x3_reduce'.format(name))
        deep_param_img['{}_3x3_reduce'.format(name)] = [kernel, biases]
        #train_layers += [kernel, biases]

    # Conv3: Output 64, pad 1, kernel 3, stride 1
    input_channels = int(conv3_r.get_shape()[-1])
    name_conv3 = '{}_3x3'.format(name)
    with tf.variable_scope(name_conv3, reuse=tf.AUTO_REUSE) as scope:
        #kernel = tf.Variable(net_data['conv3_W'], name='weights')
        kernel = tf.get_variable('weights', shape=[3, 3, input_channels, conv3_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('biases', shape=[conv3_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv3_r, kernel, [1, 1, 1, 1], padding='SAME')
        #biases = tf.Variable(net_data['conv3_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(out, name='{}_3x3'.format(name))
        deep_param_img['{}_3x3'.format(name)] = [kernel, biases]
        #train_layers += [kernel, biases]

    # Conv5_reduce: Output 64, pad 1, kernel 3, stride 1
    input_channels = int(x.get_shape()[-1])
    name_conv5r = '{}_5x5_reduce'.format(name)
    with tf.variable_scope(name_conv5r, reuse=tf.AUTO_REUSE) as scope:
        #kernel = tf.Variable(net_data['conv1_W'], name='weights')
        kernel = tf.get_variable('weights', shape=[1, 1, input_channels, conv5_r_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('biases', shape=[conv5_r_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        #biases = tf.Variable(net_data['conv1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_r = tf.nn.relu(out, name='{}_5x5_reduce'.format(name))
        deep_param_img['{}_5x5_reduce'.format(name)] = [kernel, biases]
        #train_layers += [kernel, biases]

    # Conv5: Output 64, pad 1, kernel 3, stride 1
    input_channels = int(conv5_r.get_shape()[-1])
    name_conv5 = '{}_5x5'.format(name)
    with tf.variable_scope(name_conv5, reuse=tf.AUTO_REUSE) as scope:
        #kernel = tf.Variable(net_data['conv1_W'], name='weights')
        kernel = tf.get_variable('weights', shape=[5, 5, input_channels, conv5_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('biases', shape=[conv5_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv5_r, kernel, [1, 1, 1, 1], padding='SAME')
        #biases = tf.Variable(net_data['conv1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(out, name='{}_5x5'.format(name))
        deep_param_img['{}_5x5'.format(name)] = [kernel, biases]
        #train_layers += [kernel, biases]

    pool = tf.nn.max_pool2d(x,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 1, 1, 1],
                           padding='SAME',
                           name='{}_pool'.format(name))

    input_channels = int(pool.get_shape()[-1])
    name_proj = '{}_pool_proj'.format(name)
    with tf.variable_scope(name_proj, reuse=tf.AUTO_REUSE) as scope:
        #kernel = tf.Variable(net_data['conv1_W'], name='weights')
        kernel = tf.get_variable('weights', shape=[1, 1, input_channels, pool_proj_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('biases', shape=[pool_proj_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(pool, kernel, [1, 1, 1, 1], padding='SAME')
        #biases = tf.Variable(net_data['conv1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        pool_proj = tf.nn.relu(out, name='{}_pool_proj'.format(name))
        deep_param_img['{}_pool_proj'.format(name)] = [kernel, biases]
        train_inception_layers += [kernel, biases]
        #train_last_layer += [kernel, biases]

    concate = tf.concat([conv1, conv3, conv5, pool_proj], axis=3, name='{}_concat'.format(name))
    
    return concate, deep_param_img, train_inception_layers

def non_local_block(x, num_frame, dimension, name='nll'):
    '''    Parameters:
           x         - input
           num_frame - 
           dimension - 1, 2, 3
           theta     - original size
           phi       - half size
           g         - half size
    '''
    
    batch_size, x_h, x_w, x_c = x.size
    # ConvNd
    if dimension == 3: # [batch_size, depth, input_height, input_width, channels]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[num_frame, crop_h, crop_w, 3, conv3_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            biases = tf.get_variable('biases', shape=[conv3_filters], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv3d(conv3_r, kernel, [1, num_frame, 3, 3, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            #deep_param_img['conv'] = [kernel, biases]
        theta_x = out
        maxpool = tf.nn.max_pool3d(out,
                                   ksize=[1, num_frame, 3, 3, 1],
                                   strides=[1, 1, 1, 1, 1],
                                   padding='SAME',
                                   name='{}_pool'.format(name))
        g_x = maxpool
        phi_x = maxpool
        '''        
        batchnorm = tf.contrib.layers.batch_norm(maxpool,
                                                 data_format='NHWC', # Matching the cnn tensor which has shape (?, num_frame, H, W, C, filters)
                                                 center=True,
                                                 scale=True,
                                                 is_training=training,
                                                 name='{}_bn'.format(name))
        '''
    elif dimension == 2:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[crop_h, crop_w, 3, conv3_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            biases = tf.get_variable('biases', shape=[conv3_filters], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(conv3_r, kernel, [1, 3, 3, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            theta_x = out
            #deep_param_img['conv'] = [kernel, biases]

        maxpool = tf.nn.max_pool2d(out,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='{}_pool'.format(name))
        g_x = maxpool
        phi_x = maxpool

    elif dimension == 1: # guess, pending
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('weights', shape=[crop_w, 3, conv3_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            biases = tf.get_variable('biases', shape=[conv3_filters], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv1d(conv3_r, kernel, [1, 3, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            theta_x = out
            #deep_param_img['conv'] = [kernel, biases]

        maxpool = tf.nn.max_pool1d(out,
                                   ksize=[1, 3, 1],
                                   strides=[1, 1, 1],
                                   padding='SAME',
                                   name='{}_pool'.format(name))
        g_x = maxpool
        phi_x = maxpool
        
    else:
        raise Exception('dimension should not exceed 3. The value of dimension was: {}.'.format(dimension))

    #relu = tf.nn.relu(batchnorm, name='{}_relu'.format(name))
    
    # phi, g - half spatial size
    g_channels = theta_channels // 2
    
    g_x = tf.reshape(relu, [batch_size, 3, -1], name='{}_g'.format(name))
    g_x = tf.transpose(g_x, [0,2,1])
    
    theta_x = tf.reshape(relu, [batch_size, 3, -1], name='{}_theta'.format(name))
    theta_x = tf.transpose(theta_x, [0,2,1])
    
    phi_x = tf.reshape(relu, [batch_size, 3, -1], name='{}_phi'.format(name))
    phi_x = tf.transpose(phi_x, [0,2,1])
    # before matmul: TxHxWx512 -> THWx512
    
    theta_phi = tf.matmul(theta_x, phi_x)
    theta_phi = tf.nn.softmax(theta_phi, -1)
    y = tf.matmul(theta_phi, g_x)
    y = tf.reshape(y, [batch_size, x_h, x_w, x_c])
    
    with tf.variable_scope('w', reuse=tf.AUTO_REUSE) as scope:
        w_y = tf.nn.conv2d(y, kernel, [1, 3, 3, 1], padding='SAME')
        out = tf.nn.bias_add(w_y, biases)
        if is_bn:
            w_y = tf.contrib.layers.batch_norm(w_y,
                                               data_format='NHWC', # Matching the cnn tensor which has shape (?, num_frame, H, W, C, filters)
                                               center=True,
                                               scale=True,
                                               is_training=training,
                                               name='{}_bn'.format(name))
    z = w_y + x # fc following dropout - Ouput 4096

    return z
        
def txt_mlp_layers(txt, txt_dim, output_dim, stage, model_weights=None, with_tanh=True):
    deep_param_txt = {}
    train_layers = []
    train_last_layer = []

    if model_weights is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_weights = os.path.join(
            dir_path, "pretrained_model/reference_pretrain.npy")

    net_data = dict(np.load(model_weights, encoding='bytes').item())

    # txt_fc1
    with tf.name_scope('txt_fc1'):
        if 'txt_fc1' not in net_data:
            txt_fc1w = tf.Variable(tf.truncated_normal([txt_dim, 4096],
                                                       dtype=tf.float32,
                                                       stddev=1e-2), name='weights')
            txt_fc1b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                   trainable=True, name='biases')
        else:
            txt_fc1w = tf.Variable(net_data['txt_fc1'][0], name='weights')
            txt_fc1b = tf.Variable(net_data['txt_fc1'][1], name='biases')
        txt_fc1l = tf.nn.bias_add(tf.matmul(txt, txt_fc1w), txt_fc1b)

        txt_fc1 = tf.cond(stage > 0, lambda: tf.nn.relu(
            txt_fc1l), lambda: tf.nn.dropout(tf.nn.relu(txt_fc1l), 0.5))

        train_layers += [txt_fc1w, txt_fc1b]
        deep_param_txt['txt_fc1'] = [txt_fc1w, txt_fc1b]

    # txt_fc2
    with tf.name_scope('txt_fc2'):
        if 'txt_fc2' not in net_data:
            txt_fc2w = tf.Variable(tf.truncated_normal([4096, output_dim],
                                                       dtype=tf.float32,
                                                       stddev=1e-2), name='weights')
            txt_fc2b = tf.Variable(tf.constant(0.0, shape=[output_dim], dtype=tf.float32),
                                   trainable=True, name='biases')
        else:
            txt_fc2w = tf.Variable(net_data['txt_fc2'][0], name='weights')
            txt_fc2b = tf.Variable(net_data['txt_fc2'][1], name='biases')

        txt_fc2l = tf.nn.bias_add(tf.matmul(txt_fc1, txt_fc2w), txt_fc2b)
        if with_tanh:
            txt_fc2 = tf.nn.tanh(txt_fc2l)
        else:
            txt_fc2 = txt_fc2l

        train_layers += [txt_fc2w, txt_fc2b]
        train_last_layer += [txt_fc2w, txt_fc2b]
        deep_param_txt['txt_fc2'] = [txt_fc2w, txt_fc2b]

    # return the output of text layer
    return txt_fc2, deep_param_txt, train_layers, train_last_layer

def img_vggnet_layers(img, batch_size, output_dim, stage, model_weights, with_tanh=True, val_batch_size=32):
    deep_param_img = {}
    train_layers = []
    train_last_layer = []
    print("loading image model from %s" % model_weights)
    net_data = np.load(model_weights, encoding='bytes')
    #print(net_data.files)

    # reshape bgr image to rgb image
    reshaped_image = tf.cast(img, tf.float32)[:, :, :, ::-1]

    crop_h, crop_w = (227, 227)
    # randomly crop image section
    with tf.name_scope('preprocess'):
        def train_fn():
            return tf.stack([tf.random_crop(tf.image.random_flip_left_right(each), [crop_h, crop_w, 3])
                             for each in tf.unstack(reshaped_image, batch_size)])
            
        def val_fn():
            unstacked = tf.unstack(reshaped_image, val_batch_size)
            #crop_img = tf.image.crop_to_bounding_box(img, x, y, width, height)
            
            def crop(img, x, y): return tf.image.crop_to_bounding_box(
                img, x, y, crop_w, crop_h)

            def distort(f, x, y): return tf.stack(
                [crop(f(each), x, y) for each in unstacked])

            def distort_raw(x, y): return distort(lambda x: x, x, y)

            def distort_fliped(x, y): return distort(
                tf.image.flip_left_right, x, y)
            distorted = tf.concat([distort_fliped(0, 0), distort_fliped(28, 0),
                                   distort_fliped(
                                       0, 28), distort_fliped(28, 28),
                                   distort_fliped(14, 14), distort_raw(0, 0),
                                   distort_raw(28, 0), distort_raw(0, 28),
                                   distort_raw(28, 28), distort_raw(14, 14)], 0)

            return distorted
            
    distorted = tf.cond(stage>0, val_fn, train_fn)
    
    # zero-mean input
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img-mean')
    
    distorted = distorted - mean

    # Conv1: Output 64, pad 1, kernel 3, stride 1
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(net_data['conv1_1_W'], name='weights')
        conv = tf.nn.conv2d(distorted, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv1_1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv1_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(net_data['conv1_2_W'], name='weights')
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv1_2_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv1_2'] = [kernel, biases]
        train_layers += [kernel, biases]
    #print('*', conv1_2.get_shape())
    # Pooling1
    pool1 = tf.nn.max_pool(conv1_2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    # Local response norm
    lrn1 = tf.nn.local_response_normalization(pool1,
                                              depth_radius=2,
                                              alpha=2e-05,
                                              beta=0.75,
                                              bias=1.0)
    
    # Conv2: Output: 128
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(net_data['conv2_1_W'], name='weights')
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv2_1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv2_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(net_data['conv2_2_W'], name='weights')
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv2_2_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv2_2'] = [kernel, biases]
        train_layers += [kernel, biases]
    #print('*', conv2_2.get_shape())
    # Pooling2
    pool2 = tf.nn.max_pool(conv2_2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')

    # Local response norm
    lrn2 = tf.nn.local_response_normalization(pool2,
                                              depth_radius=2,
                                              alpha=2e-05,
                                              beta=0.75,
                                              bias=1.0)
    
    # Conv3: Output 256
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(net_data['conv3_1_W'], name='weights')
        conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3_1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(net_data['conv3_2_W'], name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3_2_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(net_data['conv3_3_W'], name='weights')
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3_3_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3_3'] = [kernel, biases]
        train_layers += [kernel, biases]
    #print('*', conv3_3.get_shape())
    # Pooling3
    pool3 = tf.nn.max_pool(conv3_3,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME', # 20190423
                           name='pool3')

    # Local response norm
    lrn3 = tf.nn.local_response_normalization(pool3,
                                              depth_radius=2,
                                              alpha=2e-05,
                                              beta=0.75,
                                              bias=1.0)
    # Conv4: Output 512
    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(net_data['conv4_1_W'], name='weights')
        conv = tf.nn.conv2d(lrn3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv4_1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(net_data['conv4_2_W'], name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv4_2_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(net_data['conv4_3_W'], name='weights')
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv4_3_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4_3'] = [kernel, biases]
        train_layers += [kernel, biases]
    #print('*', conv4_3.get_shape())
    # Pooling4
    pool4 = tf.nn.max_pool(conv4_3,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME', # 20190423
                           name='pool4')
    
    # Local response norm
    lrn4 = tf.nn.local_response_normalization(pool4,
                                              depth_radius=2,
                                              alpha=2e-05,
                                              beta=0.75,
                                              bias=1.0)
    
    # Conv5: Output 512
    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(net_data['conv5_1_W'], name='weights')
        conv = tf.nn.conv2d(lrn4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv5_1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(net_data['conv5_2_W'], name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv5_2_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(net_data['conv5_3_W'], name='weights')
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv5_3_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5_3'] = [kernel, biases]
        train_layers += [kernel, biases]
    print('*', conv5_3.get_shape())
    # Pooling5
    pool5 = tf.nn.max_pool(conv5_3,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME', # 20190423
                           name='pool5')
    #print('*', pool5.get_shape())
    # FC6: Ouput (25088, 4096)
    with tf.name_scope('fc6'):
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc6w = tf.Variable(net_data['fc6_W'], name='weights')
        #print('**', shape, fc6w.get_shape()) # ** 18432 (25088, 4096)
        fc6b = tf.Variable(net_data['fc6_b'], name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6 = tf.nn.relu(fc6l)
        fc6 = tf.cond(stage > 0, lambda: fc6, lambda: tf.nn.dropout(fc6, 0.5))
        fc6o = tf.nn.relu(fc6l)
        deep_param_img['fc6'] = [fc6w, fc6b]
        train_layers += [fc6w, fc6b]
        #print('**', fc6.get_shape())
    # FC7: Output 4096
    with tf.name_scope('fc7'):
        fc7w = tf.Variable(net_data['fc7_W'], name='weights')
        fc7b = tf.Variable(net_data['fc7_b'], name='biases')
        fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7l)
        fc7 = tf.cond(stage > 0, lambda: fc7, lambda: tf.nn.dropout(fc7, 0.5))
        deep_param_img['fc7'] = [fc7w, fc7b]
        train_layers += [fc7w, fc7b]
        #print('**', fc7.get_shape())
    # FC8: Output output_dim
    with tf.name_scope('fc8'):
        # Differ train and val stage by 'fc8' as key
        if 'fc8' in net_data:
            fc8w = tf.Variable(net_data['fc8_W'], name='weights')
            fc8b = tf.Variable(net_data['fc8_b'], name='biases')
        else:
            fc8w = tf.Variable(tf.random_normal([4096, output_dim],
                                                dtype=tf.float32,
                                                stddev=1e-2), name='weights')
            fc8b = tf.Variable(tf.constant(0.0, shape=[output_dim],
                                           dtype=tf.float32), name='biases')
        fc8l = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)
        if with_tanh:
            fc8_t = tf.nn.tanh(fc8l)
        else:
            fc8_t = fc8l

        def val_fn1():
            concated = tf.concat([tf.expand_dims(i, 0)
                                  for i in tf.split(fc8_t, 10, 0)], 0)
            return tf.reduce_mean(concated, 0)
        fc8 = tf.cond(stage > 0, val_fn1, lambda: fc8_t)
        deep_param_img['fc8'] = [fc8w, fc8b]
        train_last_layer += [fc8w, fc8b]
        #print('**', fc8.get_shape()) #('**', TensorShape([Dimension(None), Dimension(64)]))
    
    print("VGG16 model loading finished")
    # Return outputs
    return fc8, deep_param_img, train_layers, train_last_layer
