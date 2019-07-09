import os
import tensorflow as tf
#from tensorflow.python.ops import array_ops
import numpy as np
import cv2 as cv

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
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[
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

def discriminator_waveone_layers():
    input_len = 128
    target = tf.placeholder(input_len, input_len, 3)
    reconstruction = tf.placeholder(input_len, input_len, 3)

    alpha = 0.2
    kernel_size = 4
    strides = 2

    x = tf.concat(values=[target, reconstruction], axis=0)
    # Conv2 - BN - leakyReLU - pending
    f1 = tf.nn.leaky_relu(x, alpha=alpha)
    f1 = tf.nn.leaky_relu(x, alpha=alpha)
    f1 = tf.nn.leaky_relu(x, alpha=alpha)
    f1 = tf.nn.leaky_relu(x, alpha=alpha)
    # conv, flatten, and dense
    
    return discriminate
    
#def spatial_rate_control():
    

def adaptive_codelength_regularization(weight_matrix):
    ''' ACR defines the 
    '''
    alpha = 0.01
    x = np.round(32 * weight_matrix + 0.5) / 32
    target_len = tf.log(np.sum(np.abs(x))) # target num of effective bits
    den = tf.log(tf.constant(10, dtype=target_len.dtype))
    code_len = target_len / den
    return alpha * code_len / (c * h * w)

#def bit_compression():
    

def optical_flow_estimator(x1, x2, LAMBDA):
    '''G(.) denotes the state-to-frame module
       return propagation
    '''
    window_size = 32
    tau = 1 # theshold
    est_flow = None
    return est_flow

def generator_waveone_layers(x, STConv_1, SIB_3c, SIB_4f, SIB_5c):
    ''' Input of generator G(.)
        pyramidal_fea - feature vectors from a single frame
        updated_state - intuitively corresponds to some prior memory (learnable)
        Return
        delta_t = x_t - m_t
        flow^hat_t
    '''
    # the generator in the paper
    train_layers = []
    deep_enc_param = {}
    motion_mode = 'optical_flow'
    input_len = 256
    in_c = 32 # just to fit in the function
    f_ch = 32 # F(.,.) function param # filter
    stride = 1
    alpha = 0.2
    # G(.) function param # pyramidal decomposition & interscale 
    out_c = 16
    out_shape = 16
    h = 16
    w =16
    g1_k_size = input_len - 1 - h
    g2_k_size = int(input_len/2) - 1 - 1 - h
    g3_k_size = int((int(input_len / 2) - 1) / 2) - 1 - 1 - h
    g4_k_size = int(int(int((int(input_len / 2) - 1) / 2) - 1) / 2) - 1 - 1 - h
    g5_k_size = h - ( int((int((int((int(input_len / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1 - 1 - 2)
    g6_k_size = h - ( int((int((int((int((int(input_len / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1 - 1 - 2)
    g7_k_size = h - ( int((int((int((int((int((int(input_len / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1 - 1)
    g8_k_size = h - ( int((int((int((int((int((int((int(input_len / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1 - 1)
    
    print(g1_k_size, g2_k_size, g3_k_size, g4_k_size, g5_k_size, g6_k_size, g7_k_size, g8_k_size)
    # build Pyramidal for coefficient extraction as tensor
    # adaptive arithmetic coding # s = AAC_Encode(b) \in {0, 1}
    
    #x1, x1_layer = separable_layer(x, in_c, 3, 4, 1)  
    x1, x1_layer = separable_layer(x, in_c, out_c, 3, stride, 'SAME', 'gen_x1')
    f1 = tf.nn.leaky_relu(x1, alpha=alpha) # return tf.nn.relu(x) - alpha*tf.nn.relu(-x)
    g1, g1_layer = separable_layer(f1, in_c, out_c, g1_k_size+2, stride, 'VALID', 'gen_g1')
    #print(x.get_shape(), x1.get_shape(), f1.get_shape(), g1.get_shape(), out_c, g1_k_size, stride, 'this is not fun')
    
    x2, x2_layer = separable_layer(STConv_1, in_c, f_ch, 3, stride, 'SAME','gen_x2')
    f2 = tf.nn.leaky_relu(x2, alpha=alpha)
    g2, g2_layer = separable_layer(f2, in_c, out_c, g2_k_size+3, stride, 'VALID', 'gen_g2')
    
    x3, x3_layer = separable_layer(SIB_3c, in_c, f_ch, 3, stride, 'SAME', 'gen_x3')
    f3 = tf.nn.leaky_relu(x3, alpha=alpha)
    g3, g3_layer = separable_layer(f3, in_c, out_c, 17, stride, 'VALID', 'gen_g3')

    x4, x4_layer = separable_layer(SIB_4f, in_c, f_ch, 3, stride, 'SAME', 'gen_x4')
    f4 = tf.nn.leaky_relu(x4, alpha=alpha)
    g4, g4_layer = separable_layer(f4, in_c, out_c, 1, stride, 'VALID', 'gen_g4')

    # build interscale alignment for multi-scale information leverage
    # learnabel upsampling using ConvXDTranspose
    #x5, x5_layer = separable_layer(x4, in_c, 3, 4, 2) # Downsampling
    x5, x5_layer = separable_layer(SIB_5c, in_c, f_ch, 3, stride, 'SAME', 'gen_x5')
    f5 = tf.nn.leaky_relu(x5, alpha=alpha)
    g5, g5_layer = separable_layer_transpose(f5, in_c, out_c, out_shape, g5_k_size, stride, 'VALID', 'gen_g5')

    #x6, x6_layer = separable_layer(x5, in_c, 3, 4, 2)
    x6, x6_layer = separable_layer(x5, in_c, f_ch, 3, stride, 'VALID', 'gen_x6')
    f6 = tf.nn.leaky_relu(x6, alpha=alpha)
    g6, g6_layer = separable_layer_transpose(f6, in_c, out_c, out_shape, g6_k_size, stride, 'VALID', 'gen_g6') 

    #x7, x7_layer = separable_layer(x6, in_c, 3, 4, 2)
    x7, x7_layer = separable_layer(x6, in_c, f_ch, 3, stride, 'VALID', 'gen_x7')
    f7 = tf.nn.leaky_relu(x7, alpha=alpha)
    g7, g7_layer = separable_layer_transpose(f7, in_c, out_c, out_shape, g7_k_size, stride, 'VALID', 'gen_g7')
    
    #x8, x8_layer = separable_layer(x7, in_c, 3 ,4 ,2)
    x8, x8_layer = separable_layer(x7, in_c, f_ch, 1, stride, 'VALID', 'gen_x8')
    f8 = tf.nn.leaky_relu(x8, alpha=alpha)
    g8, g8_layer = separable_layer_transpose(f8, in_c, out_c, out_shape, g8_k_size, stride, 'VALID', 'gen_g8')
    
    # add_n
    fe = tf.add_n([g1, g2, g3, g4, g5, g6, g7, g8]) # tf.add(x, y, name)

    g = separable_layer(fe, in_c, out_c, 3, stride, 'SAME', name='gen_reg') # add regularization
    
    # quantization suitable for encoding
    # decoding - inverse operation
    g_d = separable_layer_transpose(g, in_c, out_c, 3, 1, 'SAME', name='gen_g_d')
    
    g_d8, g_d8_layer = separable_layer(g_d, in_c, f_ch, g8_k_size, stride) # pending
    g_d8, g_d8_layer = separable_layer_transpose(g_d8, in_c, 3, 1, stride)
    f_d8 = tf.nn.leaky_relu(g_d8, alpha=alpha)
    x_d8, x_d8_layer = separable_layer_transpose(f_d8, 3, 4, 2)

    g_d7, g_d7_layer = separable_layer(g_d, in_c, f_ch, g7_k_size, stride)
    g_d7, g_d7_layer = separable_layer_transpose(g_d7, in_c, 3, 3, stride)
    f_d7 = tf.nn.leaky_relu(g_d7, alpha=alpha)
    x_d8_add = tf.add([x_d8, f_d7])
    x_d7, x_d7_layer = separable_layer_transpose(x_d8_add, in_c, 3, 4, 2)

    g_d6, g_d6_layer = separable_layer(g_d, in_c, f_ch, g6_k_size, stride)
    g_d6, g_d6_layer = separable_layer_transpose(g_d6, in_c, 3, 3, stride)
    f_d6 = tf.nn.leaky_relu(g_d6, alpha=alpha)
    x_d7_add = tf.add([x_d7, f_d6])
    x_d6, x_d6_layer = separable_layer_transpose(x_d7_add, in_c, 3, 4, 2)

    g_d5, g_d5_layer = separable_layer(g_d, in_c, f_ch, g5_k_size, stride)
    g_d5, g_d5_layer = separable_layer_transpose(g_d5, in_c, 3, 3, stride)
    f_d5 = tf.nn.leaky_relu(g_d5, alpha=alpha)
    x_d6_add = tf.add([x_d6, f_d5])
    x_d5, x_d5_layer = separable_layer_transpose(x_d6_add, in_c, 3, 4, 2)

    g_d4, g_d4_layer = separable_layer(g_d, in_c, f_ch, g4_k_size, stride)
    g_d4, g_d4_layer = separable_layer_transpose(g_d4, in_c, 3, 3, stride)
    f_d4 = tf.nn.leaky_relu(g_d4, alpha=alpha)
    x_d5_add = tf.add([x_d5, f_d4])
    x_d4, x_d4_layer = separable_layer_transpose(x_d5_add, in_c, 3, 4, 2)

    g_d3, g_d3_layer = separable_layer(g_d, in_c, f_ch, g3_k_size, stride)
    g_d3, g_d3_layer = separable_layer_transpose(g_d3, in_c, 3, 3, stride)
    f_d3 = tf.nn.leaky_relu(g_d3, alpha=alpha)
    x_d4_add = tf.add([x_d4, f_d3])
    x_d3, x_d3_layer = separable_layer_transpose(x_d4_add, in_c, 3, 4, 2)

    g_d2, g_d2_layer = separable_layer(g_d, in_c, f_ch, g2_k_size, stride)
    g_d2, g_d2_layer = separable_layer_transpose(g_d2, in_c, 3, 3, stride)
    f_d2 = tf.nn.leaky_relu(g_d2, alpha=alpha)
    x_d3_add = tf.add([x_d3, f_d2])
    x_d2, x_d2_layer = separable_layer_transpose(x_d3_add, in_c, 3, 4, 2)

    g_d1, g_d1_layer = separable_layer_transpose(g_d, in_c, f_ch, g1_k_size, stride)
    g_d1, g_d1_layer = separable_layer_transpose(g_d1, in_c, 3, 3, stride)
    f_d1 = tf.nn.leaky_relu(g_d1, alpha=alpha)
    recon = tf.add([x_d2, f_d1])
    
    # Encoder
    #hidden = ae_layers.AE_Down(distorted)    
    # Binarizer - mapping between fc layer output and what?
    #binary = ae_layers.AE_Binary(hidden)
    # Decoder
    #rec_img = ae_layers.AE_Up(binary)
    return recon, g, train_layers

def temporal_layers(img, num_img,batch_size, output_dim, stage, model_weights, with_tanh=True, val_batch_size=32):
    # temporalCNN architeture
    # ref: pending
    deep_param_img = {}
    train_layers = []
    train_last_layer = []
    print("loading temporal model from %s" % model_weights)
    net_data = np.load(model_weights, encoding='bytes')
    # reshape bgr to rgb
    reshaped_image = tf.cast(img, tf.float32)[:, :, :, ::-1]

    crop_h, crop_w = (256, 256)
    
    # preprocessing
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
        #feature_dim = out_dim
        # Conv1+convGRU+convGRU+convGRU / assuming it might work
    with tf.name_scope('conv1') as scope:
        num_filters = 64
        kernel = tf.get_variable('{}/weights'.format(scope), shape=[3, 3, 3, num_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('{}/biases'.format(scope), shape=[num_filters],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(distorted, kernel, [1, 2, 2, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name='conv1')
        deep_param_img['conv1'] = [kernel, biases]
        train_layers += [kernel, biases]

    pool1 = tf.nn.max_pool2d(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')

    with tf.name_scope('convgru1') as scope:
        shape = [pool1.get_shape()[1], pool1.get_shape()[2]]
        kernel = [3, 3]
        num_filters = 256
        convgrucell1 = ConvGRUCell(shape, num_filters, kernel)
        convgru1, state1 = tf.nn.dynamic_rnn(convgrucell1, pool1, dtype=pool1.dtype)
    with tf.name_scope('convgru2') as scope:
        shape = [convgru1.get_shape()[1], convgru1.get_shape()[2]]
        kernel = [3, 3]
        num_filters = 512
        convgrucell2 = ConvGRUCell(shape, num_filters, kernel)
        convgru2 = tf.nn.dynamic_rnn(convgrucell2, convgru1, dtype=convgru1.dtype)
    with tf.name_scope('convgru3') as scope:
        shape = [convgru2.get_shape()[1], convgru2.get_shape()[2]]
        kernel = [3, 3]
        num_filters = 512
        convgrucell3 = ConvGRUCell(shape, num_filters, kernel)
        convgru3 = tf.nn.dynamic_rnn(convgrucell3, convgru2, dtype=convgru2.dtype)
    
    return convgru3, convgru2, convgru1
       
def spatial_temporal_separable_layers(img, batch_size, output_dim, stage, model_weights, with_tanh=True, val_batch_size=32):
    # 3D Inception blocks+reduce_dim model architecture
    # ref: https://github.com/qijiezhao/s3d.pytorch.git
    deep_param_img = {}
    train_layers = []
    train_last_layer = []
    modality = 'Flow'
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
    with tf.name_scope('conv1') as scope:
        kernel = tf.get_variable('{}/weights'.format(scope), shape=[3, 3, 3, num_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('{}/biases'.format(scope), shape=[num_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(distorted, kernel, [1, 5, 5, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name='conv1')
        deep_param_img['conv1'] = [kernel, biases]
        train_layers += [kernel, biases]

    pool1 = tf.nn.max_pool2d(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')
    
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
    #print('++', pool2.get_shape())
    nlb1 = non_local_block(pool2, pool2.get_shape()[3], 32, 2, name='nlb1', is_bn=False)
    #print('++', nlb1.get_shape())
    inception3a, deep_param_img, train_inception_layers = inception_layers(nlb1,
                                                                           deep_param_img,
                                                                           192, 96, 208, 16, 48, 64,
                                                                           name='inception3a')
    train_layers += train_inception_layers
    inception3b, deep_param_img, train_inception_layers = inception_layers(inception3a,
                                                                           deep_param_img,
                                                                           160, 112, 224, 24, 64, 64,
                                                                           name='inception3b')
    train_layers += train_inception_layers
    pool3 = tf.nn.max_pool2d(inception3b,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool3')
    nlb2 = non_local_block(pool3, pool3.get_shape()[3], 32, 2, name='nlb2', is_bn=False)
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
    
    avg_pool = tf.nn.avg_pool2d(inception4b,
                              ksize=[1, 6, 6, 1],
                              strides=[1, 1, 1, 1],
                              padding='SAME', # used to be 'VALID'
                              name='avg_pool')
    
    dropout = tf.nn.dropout(avg_pool, rate=0.5) # Please use `rate` instead of `keep_prob`
    #flatten = tf.contrib.layers.flatten(dropout) # before FC

    # fc5 - Output (9216, 4096)
    with tf.name_scope('fc5') as scope:
        shape = int(np.prod(dropout.get_shape()[1:]))
        fc5w = tf.get_variable('{}/weights'.format(scope), shape=[shape, 4096],
                               initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc5b = tf.get_variable('{}/biases'.format(scope), shape=[4096], initializer=tf.constant_initializer(0.0))
        pool5_flat = tf.reshape(dropout, [-1, shape])
        fc5l = tf.nn.bias_add(tf.matmul(pool5_flat, fc5w), fc5b)
        fc5 = tf.nn.relu(fc5l)
        fc5 = tf.cond(stage > 0, lambda: fc5, lambda: tf.nn.dropout(fc5, 0.5))
        fc5o = tf.nn.relu(fc5l)
        deep_param_img['fc5'] = [fc5w, fc5b]
        train_layers += [fc5w, fc5b]

    # fc6 - Output (4096, 4096)
    with tf.name_scope('fc6') as scope:
        fc6w = tf.get_variable('{}/weights'.format(scope), shape=[4096, 4096],
                               initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc6b = tf.get_variable('{}/biases'.format(scope), shape=[4096], initializer=tf.constant_initializer(0.0))
        fc6l = tf.nn.bias_add(tf.matmul(fc5, fc6w), fc6b)
        fc6 = tf.nn.relu(fc6l)
        fc6 = tf.cond(stage > 0, lambda: fc6, lambda: tf.nn.dropout(fc6, 0.5))
        deep_param_img['fc6'] = [fc6w, fc6b]
        train_layers += [fc6w, fc6b]
    
    # fc7 layer - Ouput (None, 64)
    with tf.name_scope('fc7') as scope:
        fc7w = tf.get_variable('{}/weights'.format(scope), shape=[4096, output_dim],
                               initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc7b = tf.get_variable('{}/biases'.format(scope), shape=[output_dim], initializer=tf.constant_initializer(0.0))
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

    return fc7, deep_param_img, train_layers, train_last_layer
    
def basic_layer(x, input_channels, output_channels, kernel_size, stride, name='BasicConv3d'):
    # basic convolutional 3d: [kernel_size, kernel_size, kernel_size]
    train_basic_layer = []
    input_channels = int(x.get_shape()[-1])
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable('{}/weight'.format(scope), shape=[kernel_size, kernel_size, kernel_size, input_channels, output_channels],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('{}/bias'.format(scope), shape=[output_channels],
                                   initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(input=x,
                            filter=kernel,
                            strides=[1, stride, stride, stride, 1],
                            padding='SAME',
                            data_format='NDHWC')
        out = tf.nn.bias_add(conv, biases)
        bn = tf.contrib.layers.batch_norm(out,
                                          data_format='NHWC',
                                          center=True,
                                          scale=True,
                                          is_training=True)
        conv = tf.nn.relu(bn)
        train_basic_layer += [kernel, biases]   
        print(name, conv.get_shape()) 
    return conv, train_basic_layer

def separable_layer_transpose(x, input_channels, output_channels, output_shape, kernel_size, stride, padding='SAME', name='STConv3dTranspose'):

    # spatial-conv: [1, kernel_size, kernel_size]
    # temporal-conv: [kernel_size, 1, 1]
    train_separable_layer = []
    input_channels = int(x.get_shape()[-1])
    n_s = tf.shape(x)[0] # num of samples
    input_depth = int(x.get_shape()[1])
    input_height = int(x.get_shape()[2])
    input_width = int(x.get_shape()[3])
    
    with tf.name_scope(name) as scope:
        kernel_s = tf.get_variable('{}/weight_s'.format(scope), shape=[1, kernel_size, kernel_size, output_channels, output_channels],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases_s = tf.get_variable('{}/bias_s'.format(scope), shape=[output_channels],
                                   initializer=tf.constant_initializer(0.0))
        kernel_t = tf.get_variable('{}/weight_t'.format(scope), shape=[kernel_size, 1, 1, output_channels, input_channels],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases_t = tf.get_variable('{}/bias_t'.format(scope), shape=[output_channels],
                                   initializer=tf.constant_initializer(0.0))
        conv_t = tf.nn.conv3d_transpose(x,
                                        kernel_t,
                                        output_shape=[n_s, input_depth, input_height, input_width, output_channels],
                                        strides=[1, stride, 1, 1, 1],
                                        padding=padding)
        out_t = tf.nn.bias_add(conv_t, biases_t)
        bn_t = tf.contrib.layers.batch_norm(out_t,
                                            data_format='NHWC',
                                            center=True,
                                            scale=True,
                                            is_training=True)
        conv_t = tf.nn.relu(bn_t)
        input_depth = int(conv_t.get_shape()[1])
        conv_s = tf.nn.conv3d_transpose(conv_t,
                                        kernel_s,
                                        output_shape=[n_s, input_depth, output_shape, output_shape, output_channels],
                                        strides=[1, 1, stride, stride, 1],
                                        padding=padding)
        out_s = tf.nn.bias_add(conv_s, biases_s)
        bn_s = tf.contrib.layers.batch_norm(out_s,
                                            data_format='NHWC',
                                            center=True,
                                            scale=True,
                                            is_training=True)
        conv_s = tf.nn.relu(bn_s)
        train_separable_layer += [kernel_s, biases_s]
        
    return conv_s, train_separable_layer

def separable_layer(x, input_channels, output_channels, kernel_size, stride, padding='SAME', name='STConv3d'):
    # spatial-conv: [1, kernel_size, kernel_size]
    # temporal-conv: [kernel_size, 1, 1]
    train_separable_layer = []
    input_channels = int(x.get_shape()[-1])
    
    with tf.name_scope(name) as scope:
        kernel_s = tf.get_variable('{}/weight_s'.format(scope), shape=[1, kernel_size, kernel_size, input_channels, output_channels],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases_s = tf.get_variable('{}/bias_s'.format(scope), shape=[output_channels],
                                   initializer=tf.constant_initializer(0.0))
        kernel_t = tf.get_variable('{}/weight_t'.format(scope), shape=[kernel_size, 1, 1, output_channels, output_channels],
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases_t = tf.get_variable('{}/bias_t'.format(scope), shape=[output_channels],
                                   initializer=tf.constant_initializer(0.0))
        conv_s = tf.nn.conv3d(input=x,
                              filter=kernel_s,
                              strides=[1, 1, stride, stride, 1],
                              data_format='NDHWC',
                              padding=padding)
        out_s = tf.nn.bias_add(conv_s, biases_s)
        bn_s = tf.contrib.layers.batch_norm(out_s,
                                            data_format='NHWC',
                                            center=True,
                                            scale=True,
                                            is_training=True)
        conv_s = tf.nn.relu(bn_s)
        print(x.get_shape(), conv_s.get_shape(), output_channels, kernel_size, stride)
        conv_t = tf.nn.conv3d(input=conv_s,
                              filter=kernel_t,
                              strides=[1, stride, 1, 1, 1],
                              padding='SAME',
                              data_format='NDHWC')
        out_t = tf.nn.bias_add(conv_t, biases_t)
        bn_t = tf.contrib.layers.batch_norm(out_t,
                                            data_format='NHWC',
                                            center=True,
                                            scale=True,
                                            is_training=True)
        conv_t = tf.nn.relu(bn_t)
        train_separable_layer += [kernel_s, biases_s]
        #train_separable_layer += [kernel_t, biases_t] # init every time
        # init normal conv_t weight&bias
        print(name, conv_t.get_shape())
    return conv_t, train_separable_layer

def separable_inception_layers(x, input_channels, out_branch0, out_branch1, out_branch2, out_branch3, out_bs1, out_bs2, name='SIL'):
    # build up separable inception blocks
    # x: [batch, in_depth, in_height, in_width, in_channels]
    train_inception_layers = []
    #deep_param_img = {}
    ''' create a separable Inception block '''
    stride = 1
    bc_ksize = 1
    sc_ksize = 3
    #input_channels = int(x.get_shape()[-1])
    branch0, b0_layer = basic_layer(x, input_channels, out_branch0, bc_ksize, stride, '{}_branch0'.format(name))
    
    branch1, b1_layer = basic_layer(x, input_channels, out_branch1, bc_ksize, stride, '{}_branch1'.format(name))
    branch1, b1_layer = separable_layer(branch1, out_branch1, out_bs1, sc_ksize, stride, 'SAME','{}_branch1'.format(name))

    branch2, b2_layer = basic_layer(x, input_channels, out_branch2, bc_ksize, stride, '{}_branch2'.format(name))
    branch2, b2_layer = separable_layer(branch2, out_branch2, out_bs2, sc_ksize, stride, 'SAME', '{}_branch2'.format(name))
    
    branch3 = tf.nn.max_pool3d(x, ksize=[1,3,3,3,1], strides=[1,1,1,1,1], padding='SAME', name='{}_pool'.format(name))
    branch3, b3_layer = basic_layer(branch3, input_channels, out_branch3, bc_ksize, stride, '{}_branch3'.format(name))
    
    concate = tf.concat(values=[branch0, branch1, branch2, branch3], axis=4, name='{}_concat'.format(name)) #if 2d-data:axis=3
    # logits output: ksize=[1,1,1]
    input_channels = int(concate.get_shape()[-1])
    inception, inception_layer = basic_layer(concate, input_channels, input_channels, 1, 1, '{}_inception'.format(name))
    train_inception_layers += inception_layer
    #deep_param_img['{}_inception'.format(name)] = inception_layer
    print(name, concate.get_shape())
    return inception, train_inception_layers #, deep_param_img

def S3D(x, batch_size, output_dim, stage, model_weights, with_tanh=True, val_batch_size=16):
    ''' x: (n_frame, w, h, 3)'''
    # Pending: how to return {deep_param_img} from separable_inception
    deep_param_img = {}
    train_layers = []
    train_last_layer = []
    modality = 'Flow' # 'RBGDiff'
    GOP_size = 12
    #print("loading image model from %s" % model_weights)
    #net_data = np.load(model_weights, encoding='bytes') # if pre-trained model being available
    #print(list(net_data.keys()))

    # reshape bgr image to rgb image
    reshaped_image_buff = tf.cast(x, tf.float32)[:, :, :, :, ::-1]
    # print('reshaped_image_buff', reshaped_image_buff.get_shape()) #(?, 12, 256, 256, 3)
    crop_h, crop_w = (256, 256) # crop to inception default input image size
    print(x.get_shape(), crop_h, crop_w, 'refine shape')
    # Randomly crop a [height, width] section of each image
    with tf.name_scope('preprocess'):
        def train_fn():
            '''for 3D data
            unstacked_buff = tf.unstack(reshaped_image_buff, batch_size)
            for each in unstacked_buff:
                for n in range(GOP_size):
                    each = tf.random_crop(tf.image.random_flip_left_right(each, [crop_h, crop_w, 3]))
            stacked_buff = tf.stack(each for each in unstacked_buff)
            '''
            return tf.stack([tf.random_crop(tf.image.random_flip_left_right(each), [GOP_size, crop_h, crop_w, 3])
                     for each in tf.unstack(reshaped_image_buff, batch_size)], axis=0)
            
            #return stacked_buff

        def val_fn():
            unstacked = tf.unstack(reshaped_image_buff, val_batch_size)

            def crop(img, x, y): return tf.image.crop_to_bounding_box(
                img, x, y, crop_h, crop_w)
                # (img, ymin, xmin, ymax-ymin, xmax-xmin)
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
        mean = tf.constant([104, 117, 128], dtype=tf.float32, shape=[1, 1, 1, 3], name='video-mean')
        distorted = distorted - mean
    # basic_layer(x, input_channels, output_channels, kernel_size, stride, name='BasicConv3d')
    # separable_layer(x, input_channels, output_channels, kernel_size, stride, name='STConv3d')
    # separable_inception_layers(x, in_channels, out_branch0, out_branch1, out_branch2, out_branch3, out_bs1, out_bs2, name)
    STConv_1, STConv1_layer = separable_layer(distorted, 3, 64, 7, 2, 'SAME', name='STConv_1')
    Pool_1 = tf.nn.max_pool3d(STConv_1, ksize=[1,1,3,3,1], strides=[1,1,2,2,1], padding='SAME', name='Pool_1')
    deep_param_img['STConv_1'] = STConv1_layer
    train_layers += STConv1_layer
    
    BConv_2, BConv2_layer = basic_layer(Pool_1, 64, 64, 1, 1, name='BConv_1')
    deep_param_img['BConv_2'] = BConv2_layer
    train_layers += BConv2_layer
    
    STConv_2, STConv2_layer = separable_layer(BConv_2, 64, 192, 3, 1, 'SAME', name='STConv_2')
    Pool_2 = tf.nn.max_pool3d(STConv_2, ksize=[1,1,3,3,1], strides=[1,1,2,2,1], padding='SAME', name='Pool_2')
    deep_param_img['STConv_2'] = STConv2_layer
    train_layers += STConv2_layer
    
    Mixed_3b, SIB3b_layer = separable_inception_layers(Pool_2, 256, 64, 96, 16, 32, 128, 32, name='SIB_3b')
    deep_param_img['SIB_3b'] = SIB3b_layer
    train_layers += SIB3b_layer
    Mixed_3c, SIB3c_layer = separable_inception_layers(Mixed_3b, 256, 128, 128, 32, 64, 192, 96, name='SIB_3c')
    Pool_3 = tf.nn.max_pool3d(Mixed_3c, ksize=[1,3,3,3,1], strides=[1,2,2,2,1], padding='SAME', name='Pool_3')
    deep_param_img['SIB_3c'] = SIB3c_layer
    train_layers += SIB3c_layer
    
    Mixed_4b, SIB4b_layer = separable_inception_layers(Pool_3, 256, 192, 96, 16, 64, 208, 48, name='SIB_4b')
    deep_param_img['SIB_4b'] = SIB4b_layer
    train_layers += SIB4b_layer
    Mixed_4c, SIB4c_layer = separable_inception_layers(Mixed_4b, 256, 160, 112, 24, 64, 224, 64, name='SIB_4c')
    deep_param_img['SIB_4c'] = SIB4c_layer
    train_layers += SIB4c_layer
    Mixed_4d, SIB4d_layer = separable_inception_layers(Mixed_4c, 512, 128, 128, 24, 64, 256, 64, name='SIB_4d')
    deep_param_img['SIB_4d'] = SIB4d_layer
    train_layers += SIB4d_layer
    Mixed_4e, SIB4e_layer = separable_inception_layers(Mixed_4d, 512, 112, 144, 32, 64, 288, 64, name='SIB_4e')
    deep_param_img['SIB_4e'] = SIB4e_layer
    train_layers += SIB4e_layer
    Mixed_4f, SIB4f_layer = separable_inception_layers(Mixed_4e, 528, 256, 160, 32, 128, 320, 128, name='SIB_4f')
    deep_param_img['SIB_4f'] = SIB4f_layer
    train_layers += SIB4f_layer
    Pool_4 =tf.nn.max_pool3d(Mixed_4f, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name='Pool_4')
    
    Mixed_5b, SIB5b_layer = separable_inception_layers(Pool_4, 832, 256, 160, 32, 128, 320, 128, name='SIB_5b')
    deep_param_img['SIB_5b'] = SIB5b_layer
    train_layers += SIB5b_layer    
    Mixed_5c, SIB5c_layer = separable_inception_layers(Mixed_5b, 832, 384, 192, 48, 128, 384, 128, name='SIB_5c')
    deep_param_img['SIB_5c'] = SIB5c_layer
    train_layers += SIB5c_layer
    Pool_5 = tf.nn.avg_pool3d(Mixed_5c, ksize=[1,2,7,7,1], strides=[1,1,1,1,1], padding='SAME', name='Pool_5')
    
    Dropout = tf.nn.dropout(Pool_5, rate=0.3) # rate \in [0, 1) follows the S3DG setting
    '''
    # fc6 - output (131072, 4096) # 131072=2*8*8*1024
    with tf.name_scope('FC_6') as scope:
        shape = int(np.prod(Dropout.get_shape()[1:]))
        fc6w = tf.get_variable('{}/weights'.format(scope), shape=[shape, 4096],
                               initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc6b = tf.get_variable('{}/biases'.format(scope), shape=[4096], initializer=tf.constant_initializer(0.0))
        pool5_flat = tf.reshape(Dropout, [-1, shape])
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6 = tf.nn.relu(fc6l)
        fc6 = tf.cond(stage > 0, lambda: fc6, lambda: tf.nn.dropout(fc6, 0.5))
        fc6o = tf.nn.relu(fc6l)
        deep_param_img['FC_6'] = [fc6w, fc6b]
        train_layers += [fc6w, fc6b]

    # fc7 - Output (4096, 4096)
    with tf.name_scope('FC_7') as scope:
        fc7w = tf.get_variable('{}/weights'.format(scope), shape=[4096, 4096],
                               initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc7b = tf.get_variable('{}/biases'.format(scope), shape=[4096], initializer=tf.constant_initializer(0.0))
        fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7l)
        fc7 = tf.cond(stage > 0, lambda: fc7, lambda: tf.nn.dropout(fc7, 0.5))
        deep_param_img['FC_7'] = [fc7w, fc7b]
        train_layers += [fc7w, fc7b]

    # fc8 - Ouput (None, 64)
    with tf.name_scope('FC_8') as scope:
        fc8w = tf.get_variable('{}/weights'.format(scope), shape=[4096, output_dim],
                               initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc8b = tf.get_variable('{}/biases'.format(scope), shape=[output_dim], initializer=tf.constant_initializer(0.0))
        fc8l = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)
        if with_tanh:
            fc8_t = tf.nn.tanh(fc8l)
        else:
            fc8_t = fc8l
        def val_fn1():
            concated = tf.concat([tf.expand_dims(i, 0) for i in tf.split(fc8_t, 10, 0)], 0)
            return tf.reduce_mean(concated, 0)
        fc8 = tf.cond(stage > 0, val_fn1, lambda: fc8_t)
        deep_param_img['FC_8'] = [fc8w, fc8b]
        train_last_layer += [fc8w, fc8b]# input_channels_fc_6: 1024
    '''
    recon, bits, generator_layers = generator_waveone_layers(distorted, STConv_1, Mixed_3c, Mixed_4f, Mixed_5c)
    print("S3D Model Loading Done") 
    return fc8, deep_param_img, train_layers, train_last_layer
    
def inception_layers(x, deep_param_img, conv1_filters, conv3_r_filters, conv3_filters, conv5_r_filters, conv5_filters, pool_proj_filters, name='inception'):
    # frame_stride=4 # RGB_diff: optical_flow
    # ref: https://github.com/Natsu6767/Inception-Module-Tensorflow
    train_inception_layers = []
    ''' create an Inception layer '''
    # Conv1: Output 64, pad 1, kernel 3, stride 1
    input_channels = int(x.get_shape()[-1])
    name_conv1 = '{}_1x1'.format(name)
    with tf.name_scope(name_conv1) as scope:
        kernel = tf.get_variable('{}/weights'.format(scope), shape=[1, 1, input_channels, conv1_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('{}/biases'.format(scope), shape=[conv1_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name='{}_1x1'.format(name))
        deep_param_img['{}_1x1'.format(name)] = [kernel, biases]
    
    # Conv3_reduce: Output 64, pad 1, kernel 3, stride 1
    input_channels = int(x.get_shape()[-1])
    name_conv3r = '{}_3x3_reduce'.format(name)
    with tf.name_scope(name_conv3r) as scope:
        kernel = tf.get_variable('{}/weights'.format(scope), shape=[1, 1, input_channels, conv3_r_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('{}/biases'.format(scope), shape=[conv3_r_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv3_r = tf.nn.relu(out, name='{}_3x3_reduce'.format(name))
        deep_param_img['{}_3x3_reduce'.format(name)] = [kernel, biases]

    # Conv3: Output 64, pad 1, kernel 3, stride 1
    input_channels = int(conv3_r.get_shape()[-1])
    name_conv3 = '{}_3x3'.format(name)
    with tf.name_scope(name_conv3) as scope:
        kernel = tf.get_variable('{}/weights'.format(scope), shape=[3, 3, input_channels, conv3_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('{}/biases'.format(scope), shape=[conv3_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv3_r, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(out, name='{}_3x3'.format(name))
        deep_param_img['{}_3x3'.format(name)] = [kernel, biases]

    # Conv5_reduce: Output 64, pad 1, kernel 3, stride 1
    input_channels = int(x.get_shape()[-1])
    name_conv5r = '{}_5x5_reduce'.format(name)
    with tf.name_scope(name_conv5r) as scope:
        kernel = tf.get_variable('{}/weights'.format(scope), shape=[1, 1, input_channels, conv5_r_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('{}/biases'.format(scope), shape=[conv5_r_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv5_r = tf.nn.relu(out, name='{}_5x5_reduce'.format(name))
        deep_param_img['{}_5x5_reduce'.format(name)] = [kernel, biases]

    # Conv5: Output 64, pad 1, kernel 3, stride 1
    input_channels = int(conv5_r.get_shape()[-1])
    name_conv5 = '{}_5x5'.format(name)
    with tf.name_scope(name_conv5) as scope:
        kernel = tf.get_variable('{}/weights'.format(scope), shape=[5, 5, input_channels, conv5_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('{}/biases'.format(scope), shape=[conv5_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv5_r, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(out, name='{}_5x5'.format(name))
        deep_param_img['{}_5x5'.format(name)] = [kernel, biases]

    pool = tf.nn.max_pool2d(x,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 1, 1, 1],
                           padding='SAME',
                           name='{}_pool'.format(name))

    input_channels = int(pool.get_shape()[-1])
    name_proj = '{}_pool_proj'.format(name)
    with tf.name_scope(name_proj) as scope:
        kernel = tf.get_variable('{}/weights'.format(scope), shape=[1, 1, input_channels, pool_proj_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('{}/biases'.format(scope), shape=[pool_proj_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(pool, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        pool_proj = tf.nn.relu(out, name='{}_pool_proj'.format(name))
        deep_param_img['{}_pool_proj'.format(name)] = [kernel, biases]
        train_inception_layers += [kernel, biases]

    concate = tf.concat([conv1, conv3, conv5, pool_proj], axis=3, name='{}_concat'.format(name))
    
    return concate, deep_param_img, train_inception_layers

def non_local_block(x, k_c, num_frame, dimension, name='nlb', is_bn=False):
    '''    Parameters:
           x         - input
           k_c       - output channels
           num_frame - for conv3d (dim=3) only
           dimension - 1, 2, 3
           theta     - original size
           phi       - half size
           g         - half size
    '''
    batch_size = tf.shape(x)[0]
    x_h = tf.shape(x)[1]
    x_w = tf.shape(x)[2]
    x_c = x.get_shape()[3]      # for unknown reasons
    # ConvNd
    if dimension == 3: # [batch_size, depth, input_height, input_width, channels]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('{}/weights'.format(name), shape=[num_frame, 1, 1, x_c, k_c],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            biases = tf.get_variable('{}/biases'.format(name), shape=[k_c], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv3d(x, kernel, [1, num_frame, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            
        theta_x = out
        maxpool = tf.nn.max_pool3d(out,
                                   ksize=[1, num_frame, 2, 2, 1],
                                   strides=[1, 1, 2, 2, 1],
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
            kernel = tf.get_variable('{}/weights'.format(name), shape=[1, 1, x_c, k_c],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            biases = tf.get_variable('{}/biases'.format(name), shape=[k_c], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            theta_x = out

        maxpool = tf.nn.max_pool2d(out,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 name='{}_pool'.format(name))
        g_x = maxpool
        phi_x = maxpool

    elif dimension == 1: # guess, pending
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable('{}/weights'.format(name), shape=[1, x_c, k_c],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            biases = tf.get_variable('{}/biases'.format(name), shape=[k_c], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv1d(x, kernel, [1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            theta_x = out

        maxpool = tf.nn.max_pool1d(out,
                                   ksize=[1, 2, 1],
                                   strides=[1, 2, 1],
                                   padding='SAME',
                                   name='{}_pool'.format(name))
        g_x = maxpool
        phi_x = maxpool
        
    else:
        raise Exception('dimension should not exceed 3. The value of dimension was: {}.'.format(dimension))
    
    # phi, g - half spatial size
    g_x = tf.reshape(g_x, [batch_size, k_c, -1], name='{}_g'.format(name))
    g_x = tf.transpose(g_x, [0,2,1])
    
    theta_x = tf.reshape(theta_x, [batch_size, k_c, -1], name='{}_theta'.format(name))
    theta_x = tf.transpose(theta_x, [0,2,1])
    
    phi_x = tf.reshape(phi_x, [batch_size, k_c, -1], name='{}_phi'.format(name))
    #phi_x = tf.transpose(phi_x, [0,2,1])
    # before matmul: TxHxWx512 -> THWx512
    
    theta_phi = tf.matmul(theta_x, phi_x)
    theta_phi = tf.nn.softmax(theta_phi, -1)
    
    y = tf.matmul(theta_phi, g_x)
    y = tf.reshape(y, [batch_size, x_h, x_w, k_c])
    
    with tf.variable_scope('{}_w'.format(name), reuse=tf.AUTO_REUSE) as scope:
        kernel = tf.get_variable('weights', shape=[1, 1, x_c, k_c],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        biases = tf.get_variable('biases', shape=[k_c], initializer=tf.constant_initializer(0.0))
        w_y = tf.nn.conv2d(y, kernel, [1, 1, 1, 1], padding='SAME')
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
