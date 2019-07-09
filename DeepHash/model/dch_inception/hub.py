import tensorflow as tf
import tensorflow_hub as hub
'''
Here is a mapping from the old_names to the new names:
  Old name          | New name
  =======================================
  conv0             | Conv2d_1a_3x3
  conv1             | Conv2d_2a_3x3
  conv2             | Conv2d_2b_3x3
  pool1             | MaxPool_3a_3x3
  conv3             | Conv2d_3b_1x1
  conv4             | Conv2d_4a_3x3
  pool2             | MaxPool_5a_3x3
  mixed_35x35x256a  | Mixed_5b
  mixed_35x35x288a  | Mixed_5c
  mixed_35x35x288b  | Mixed_5d
  mixed_17x17x768a  | Mixed_6a
  mixed_17x17x768b  | Mixed_6b
  mixed_17x17x768c  | Mixed_6c
  mixed_17x17x768d  | Mixed_6d
  mixed_17x17x768e  | Mixed_6e
  mixed_8x8x1280a   | Mixed_7a
  mixed_8x8x2048a   | Mixed_7b
  mixed_8x8x2048b   | Mixed_7c
'''

images = tf.placeholder(tf.float32, (None, 299, 299, 3))

module = hub.Module('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1', trainable=True) # for fine-tuning and weights modify
# feature_vector module doesn't include the final desnse classification layer
# url + '?tf-hub-format=compressed'
module_spec = hub.load_module_spec('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
height, width = hub.get_expected_image_size(module_spec)
print(height, width)
module_fea = module(dict(images=images), signature='image_feature_vector', as_dict=True)
# passing as_dict=True enables access to a whole set of intermediate activations
print(module.get_signature_names())
print(module.get_output_info_dict(signature='image_feature_vector'))

def module_fn():
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 50])
    layer1 = tf.layers.dense(inputs, 200)
    layer2 = tf.layers.dense(layer1, 100)
    outputs = dict(default=layer2, hidden_activations=layer1)
    hub.add_signature(inputs=inputs, outputs=outputs)

spec = hub.create_module_spec(module_fn)
spec.export("./export_module", checkpoint_path="./training_model")


deep_param_img = {}
train_layers = []
train_last_layer = []
global_pool = module_fea['InceptionV3/global_pool']
print(module_fea.items())


checkpoints_name = '/home/links/ym310/DeepHash/architecture/pretrained_model/inception_v3.ckpt'
model_name = 'InceptionV3'
print("loading pretrained graph checkpoint from %s" % checkpoints_name)
sess = tf.InteractiveSession()
train_graph = tf.Graph()
graph.as_default()
    
'''[(u'default', <tf.Tensor 'module_apply_image_feature_vector/hub_output/feature_vector/SpatialSqueeze:0' shape=(?, 2048) dtype=float32>), 
    (u'InceptionV3/global_pool', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/Logits/GlobalPool:0' shape=(?, 1, 1, 2048) dtype=float32>),
    (u'InceptionV3/MaxPool_3a_3x3', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/MaxPool_3a_3x3/MaxPool:0' shape=(?, 73, 73, 64) dtype=float32>),
    (u'InceptionV3/Conv2d_3b_1x1', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu:0' shape=(?, 73, 73, 80) dtype=float32>),
    (u'InceptionV3/Conv2d_1a_3x3', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu:0' shape=(?, 149, 149, 32) dtype=float32>),
    (u'InceptionV3/Mixed_7b', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Mixed_7b/concat:0' shape=(?, 8, 8, 2048) dtype=float32>), 
    (u'InceptionV3/Mixed_5b', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Mixed_5b/concat:0' shape=(?, 35, 35, 256) dtype=float32>), 
    (u'InceptionV3/Mixed_5c', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Mixed_5c/concat:0' shape=(?, 35, 35, 288) dtype=float32>), 
    (u'InceptionV3/Mixed_5d', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Mixed_5d/concat:0' shape=(?, 35, 35, 288) dtype=float32>),   
    (u'InceptionV3/Mixed_6a', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Mixed_6a/concat:0' shape=(?, 17, 17, 768) dtype=float32>), 
    (u'InceptionV3/Conv2d_2a_3x3', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu:0' shape=(?, 147, 147, 32) dtype=float32>),
    (u'InceptionV3/Mixed_6c', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Mixed_6c/concat:0' shape=(?, 17, 17, 768) dtype=float32>), 
    (u'InceptionV3/Mixed_6b', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Mixed_6b/concat:0' shape=(?, 17, 17, 768) dtype=float32>), 
    (u'InceptionV3/Mixed_6e', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Mixed_6e/concat:0' shape=(?, 17, 17, 768) dtype=float32>), 
    (u'InceptionV3/Mixed_6d', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Mixed_6d/concat:0' shape=(?, 17, 17, 768) dtype=float32>), 
    (u'InceptionV3/Mixed_7a', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Mixed_7a/concat:0' shape=(?, 8, 8, 1280) dtype=float32>), 
    (u'InceptionV3/Conv2d_4a_3x3', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu:0' shape=(?, 71, 71, 192) dtype=float32>),
    (u'InceptionV3/MaxPool_5a_3x3', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/MaxPool_5a_3x3/MaxPool:0' shape=(?, 35, 35, 192) dtype=float32>),
    (u'InceptionV3/Conv2d_2b_3x3', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu:0' shape=(?, 147, 147, 64) dtype=float32>),
    (u'InceptionV3/Mixed_7c', <tf.Tensor 'module_apply_image_feature_vector/InceptionV3/InceptionV3/Mixed_7c/concat:0' shape=(?, 8, 8, 2048) dtype=float32>)]
'''
print(global_pool)


#deep_param_img = tf.Variable(module_fea['InceptionV3/global_pool'], name='global_pool')
#train_last_layer += [tf.Variable(module_fea['InceptionV3/global_pool'], name='weights')]
#def _save_model(sess, epoch_n):
#    export_dir
#tf.saved_model.simple_save(sess, export_dir, inputs, outputs)


#[u'default', u'image_feature_vector']
#{u'InceptionV3/global_pool': <hub.ParsedTensorInfo shape=(?, 1, 1, 2048) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Mixed_5c': <hub.ParsedTensorInfo shape=(?, 35, 35, 288) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Conv2d_2a_3x3': <hub.ParsedTensorInfo shape=(?, 147, 147, 32) dtype=float32 is_sparse=False>, \\
# u'default': <hub.ParsedTensorInfo shape=(?, 2048) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Mixed_7b': <hub.ParsedTensorInfo shape=(?, 8, 8, 2048) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Mixed_5b': <hub.ParsedTensorInfo shape=(?, 35, 35, 256) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/MaxPool_3a_3x3': <hub.ParsedTensorInfo shape=(?, 73, 73, 64) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Mixed_5d': <hub.ParsedTensorInfo shape=(?, 35, 35, 288) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Mixed_6a': <hub.ParsedTensorInfo shape=(?, 17, 17, 768) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Conv2d_3b_1x1': <hub.ParsedTensorInfo shape=(?, 73, 73, 80) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Mixed_6c': <hub.ParsedTensorInfo shape=(?, 17, 17, 768) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Mixed_6b': <hub.ParsedTensorInfo shape=(?, 17, 17, 768) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Mixed_6e': <hub.ParsedTensorInfo shape=(?, 17, 17, 768) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Mixed_6d': <hub.ParsedTensorInfo shape=(?, 17, 17, 768) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Conv2d_1a_3x3': <hub.ParsedTensorInfo shape=(?, 149, 149, 32) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Mixed_7a': <hub.ParsedTensorInfo shape=(?, 8, 8, 1280) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Conv2d_4a_3x3': <hub.ParsedTensorInfo shape=(?, 71, 71, 192) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/MaxPool_5a_3x3': <hub.ParsedTensorInfo shape=(?, 35, 35, 192) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Conv2d_2b_3x3': <hub.ParsedTensorInfo shape=(?, 147, 147, 64) dtype=float32 is_sparse=False>, \\
# u'InceptionV3/Mixed_7c': <hub.ParsedTensorInfo shape=(?, 8, 8, 2048) dtype=float32 is_sparse=False>} \\

