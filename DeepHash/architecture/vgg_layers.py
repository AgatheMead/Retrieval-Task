import os
import tensorflow as tf
import numpy as np

def img_vggnet_layers(img, batch_size, output_dim, stage, model_weights, with_tanh=True, val_batch=32):
    print("loading image model from %s" % model_weights)
    net_data = np.load(model_weights, encoding='bytes')
    print(net_data.files)
''' ['conv4_3_W', 'conv5_1_b', 'conv1_2_b', 'conv5_2_b', 'conv1_1_W', 'conv5_3_b', 'conv5_2_W', 'conv5_3_W', 'conv1_1_b', 'fc7_b', 'conv5_1_W', 'conv1_2_W', 'conv3_2_W', 'conv4_2_b', 'conv4_1_b', 'conv3_3_W', 'conv2_1_b', 'conv3_1_b', 'conv2_2_W', 'fc6_b', 'fc8_b', 'conv4_3_b', 'conv2_2_b', 'fc6_W', 'fc8_W', 'fc7_W', 'conv3_2_b', 'conv4_2_W', 'conv3_3_b', 'conv3_1_W', 'conv2_1_W', 'conv4_1_W'] '''
    
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
            Dee
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
            
    distorted = tf.cond(stage>0, val_fn(), train_fn())
    
    # zero-mean input
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img-mean')
    
    distorted = distorted - mean

    # Conv1: Output 64, pad 1, kernel 3, stride 1
    with tf.name.scope('conv1_1') as scope:
        kernel = tf.Variable(net_data['conv1_1_W'], name='weights')
        conv = tf.nn.conv2d(distored, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv1_1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv1_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name.scope('conv1_2') as scope:
        kernel = tf.Variable(net_data['conv1_2_W'], name='weights')
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv1_2_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv1_2'] = [kernel, biases]
        train_layers += [kernel, biases]

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
                                              beta=0.75
                                              bias=bias)
    
    # Conv2
    with tf.name.scope('conv2_1') as scope:
        kernel = tf.Variable(net_data['conv2_1_W'], name='weights')
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv2_1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv2_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name.scope('conv2_2') as scope:
        kernel = tf.Variable(net_data['conv2_2_W'], name='weights')
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv2_2_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv2_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Pooling2
    pool2 = tf.nn.max_pool(conv2_2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    # Local response norm
    lrn2 = tf.nn.local_response_normalization(pool2,
                                              depth_radius=2,
                                              alpha=2e-05,
                                              beta=0.75
                                              bias=bias)
    
    # Conv3
    with tf.name.scope('conv3_1') as scope:
        kernel = tf.Variable(net_data['conv3_1_W'], name='weights')
        conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3_1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name.scope('conv3_2') as scope:
        kernel = tf.Variable(net_data['conv3_2_W'], name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3_2_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name.scope('conv3_3') as scope:
        kernel = tf.Variable(net_data['conv3_3_W'], name='weights')
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3_3_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3_3'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Pooling3
    pool3 = tf.nn.max_pool(conv3_3,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    # Local response norm
    lrn3 = tf.nn.local_response_normalization(pool3,
                                              depth_radius=2,
                                              alpha=2e-05,
                                              beta=0.75
                                              bias=bias)
    # Conv4
    with tf.name.scope('conv4_1') as scope:
        kernel = tf.Variable(net_data['conv4_1_W'], name='weights')
        conv = tf.nn.conv2d(lrn3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv4_1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name.scope('conv4_2') as scope:
        kernel = tf.Variable(net_data['conv4_2_W'], name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv4_2_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name.scope('conv4_3') as scope:
        kernel = tf.Variable(net_data['conv4_3_W'], name='weights')
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv4_3_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4_3'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Pooling4
    pool4 = tf.nn.max_pool(conv4_3,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    # Local response norm
    lrn4 = tf.nn.local_response_normalization(pool4,
                                              depth_radius=2,
                                              alpha=2e-05,
                                              beta=0.75
                                              bias=bias)
    
    # Conv5
    with tf.name.scope('conv5_1') as scope:
        kernel = tf.Variable(net_data['conv5_1_W'], name='weights')
        conv = tf.nn.conv2d(lrn4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv5_1_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5_1'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name.scope('conv5_2') as scope:
        kernel = tf.Variable(net_data['conv5_2_W'], name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv5_2_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5_2'] = [kernel, biases]
        train_layers += [kernel, biases]

    with tf.name.scope('conv5_3') as scope:
        kernel = tf.Variable(net_data['conv5_3_W'], name='weights')
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv5_3_b'], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5_3'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Pooling5
    pool5 = tf.nn.max_pool(conv5_3,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    # FC6: Ouput 4096
    with tf.name_scope('fc6'):
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc6w = tf.Variable(net_data['fc6_W'], name='weights')
        fc6b = tf.Variable(net_data['fc6_b'], name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6 = tf.nn.relu(fc6l)
        fc6 = tf.cond(stage > 0, lambda: fc6, lambda: tf.nn.dropout(fc6, 0.5))
        fc6o = tf.nn.relu(fc6l)
        deep_param_img['fc6'] = [fc6w, fc6b]
        train_layers += [fc6w, fc6b]

    # FC7: Output 4096
    with tf.name_scope('fc7'):
        fc7w = tf.Variable(net_data['fc7_W'], name='weights')
        fc7b = tf.Variable(net_data['fc7_b'], name='biases')
        fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7l)
        fc7 = tf.cond(stage > 0, lambda: fc7, lambda: tf.nn.dropout(fc7, 0.5))
        deep_param_img['fc7'] = [fc7w, fc7b]
        train_layers += [fc7w, fc7b]

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

    print("VGG16 model loading finished")
    # Return outputs
    return fc8, deep_param_img, train_layers, train_last_layer

def temporal_segmentation_layers():
    return 0
