from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
import sys
from datetime import datetime
import os
import scipy.misc
from PIL import Image
import six

slim = tf.contrib.slim


def crop_features(feature, out_size):
    """Crop the center of a feature map
    feature: Feature map to crop
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    """
    up_size = tf.shape(feature)
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    # slice_input = tf.slice(feature, (0, ini_w, ini_w, 0), (-1, out_size[1], out_size[2], -1))  # Caffe cropping way
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])


def osvos_rcf(inputs, scope='osvos'):
    """Defines the OSVOS network
    inputs: Tensorflow placeholder that contains the input image
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """
    im_size = tf.shape(inputs)

    with tf.variable_scope(scope, 'osvos', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs of all intermediate layers.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            outputs_collections=end_points_collection):
            net_1_1 = slim.conv2d(inputs, 64, [3, 3], scope='conv1/conv1_1')
            net_1_2 = slim.conv2d(net_1_1, 64, [3, 3], scope='conv1/conv1_2')
            net = slim.max_pool2d(net_1_2, [2, 2], scope='pool1')
            net_2_1 = slim.conv2d(net, 128, [3, 3], scope='conv2/conv2_1')
            net_2_2 = slim.conv2d(net_2_1, 128, [3, 3], scope='conv2/conv2_2')
            net = slim.max_pool2d(net_2_2, [2, 2], scope='pool2')
            # net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            # net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            # net = slim.max_pool2d(net_2, [2, 2], scope='pool2')
            net_3_1 = slim.conv2d(net, 256, [3, 3], scope='conv3/conv3_1')
            net_3_2 = slim.conv2d(net_3_1, 256, [3, 3], scope='conv3/conv3_2')
            net_3_3 = slim.conv2d(net_3_2, 256, [3, 3], scope='conv3/conv3_3')
            net = slim.max_pool2d(net_3_3, [2, 2], scope='pool3')
            net_4_1 = slim.conv2d(net, 512, [3, 3], scope='conv4/conv4_1')
            net_4_2 = slim.conv2d(net_4_1, 512, [3, 3], scope='conv4/conv4_2')
            net_4_3 = slim.conv2d(net_4_2, 512, [3, 3], scope='conv4/conv4_3')
            net = slim.max_pool2d(net_4_3, [2, 2], scope='pool4')
            net_5_1 = slim.conv2d(net, 512, [3, 3], scope='conv5/conv5_1')
            net_5_2 = slim.conv2d(net_5_1, 512, [3, 3], scope='conv5/conv5_2')
            net_5_3 = slim.conv2d(net_5_2, 512, [3, 3], scope='conv5/conv5_3')

            # net_3 = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            # net = slim.max_pool2d(net_3, [2, 2], scope='pool3')
            # net_4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            # net = slim.max_pool2d(net_4, [2, 2], scope='pool4')
            # net_5 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

            # Get side outputs of the network
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None):
                # side_1_1 = slim.conv2d(net_1_1, 16, [3, 3], scope='conv1_1_16')
                # side_1_2 = slim.conv2d(net_1_2, 16, [3, 3], scope='conv1_2_16')
                side_2_1 = slim.conv2d(net_2_1, 16, [3, 3], scope='conv2_1_16')
                side_2_2 = slim.conv2d(net_2_2, 16, [3, 3], scope='conv2_2_16')
                side_3_1 = slim.conv2d(net_3_1, 16, [3, 3], scope='conv3_1_16')
                side_3_2 = slim.conv2d(net_3_2, 16, [3, 3], scope='conv3_2_16')
                side_3_3 = slim.conv2d(net_3_3, 16, [3, 3], scope='conv3_3_16')
                side_4_1 = slim.conv2d(net_4_1, 16, [3, 3], scope='conv4_1_16')
                side_4_2 = slim.conv2d(net_4_2, 16, [3, 3], scope='conv4_2_16')
                side_4_3 = slim.conv2d(net_4_3, 16, [3, 3], scope='conv4_3_16')
                side_5_1 = slim.conv2d(net_5_1, 16, [3, 3], scope='conv5_1_16')
                side_5_2 = slim.conv2d(net_5_2, 16, [3, 3], scope='conv5_2_16')
                side_5_3 = slim.conv2d(net_5_3, 16, [3, 3], scope='conv5_3_16')
                # side_2 = slim.conv2d(net_2, 16, [3, 3], scope='conv2_2_16')
                # side_3 = slim.conv2d(net_3, 16, [3, 3], scope='conv3_3_16')
                # side_4 = slim.conv2d(net_4, 16, [3, 3], scope='conv4_3_16')
                # side_5 = slim.conv2d(net_5, 16, [3, 3], scope='conv5_3_16')

                # Supervise side outputs
                # side_1 = side_1_1 + side_1_2
                side_2 = side_2_1 + side_2_2
                side_3 = side_3_1 + side_3_2 + side_3_3
                side_4 = side_4_1 + side_4_2 + side_4_3
                side_5 = side_5_1 + side_5_2 + side_5_3

                # side_1_s = slim.conv2d(side_1, 1, [1, 1], scope='score-dsn_1')
                side_2_s = slim.conv2d(side_2, 1, [1, 1], scope='score-dsn_2')
                side_3_s = slim.conv2d(side_3, 1, [1, 1], scope='score-dsn_3')
                side_4_s = slim.conv2d(side_4, 1, [1, 1], scope='score-dsn_4')
                side_5_s = slim.conv2d(side_5, 1, [1, 1], scope='score-dsn_5')
                with slim.arg_scope([slim.convolution2d_transpose],
                                    activation_fn=None, biases_initializer=None, padding='VALID',
                                    outputs_collections=end_points_collection, trainable=False):
                    # Side outputs
                    # side_1_s = slim.convolution2d_transpose(side_2_s, 1, 2, 1, scope='score-dsn_1-up')
                    # side_1_s = crop_features(side_1_s, im_size)
                    # utils.collect_named_outputs(end_points_collection, 'osvos/score-dsn_1-cr', side_1_s)
                    side_2_s = slim.convolution2d_transpose(side_2_s, 1, 4, 2, scope='score-dsn_2-up')
                    side_2_s = crop_features(side_2_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/score-dsn_2-cr', side_2_s)
                    side_3_s = slim.convolution2d_transpose(side_3_s, 1, 8, 4, scope='score-dsn_3-up')
                    side_3_s = crop_features(side_3_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/score-dsn_3-cr', side_3_s)
                    side_4_s = slim.convolution2d_transpose(side_4_s, 1, 16, 8, scope='score-dsn_4-up')
                    side_4_s = crop_features(side_4_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/score-dsn_4-cr', side_4_s)
                    side_5_s = slim.convolution2d_transpose(side_5_s, 1, 32, 16, scope='score-dsn_5-up')
                    side_5_s = crop_features(side_5_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/score-dsn_5-cr', side_5_s)

                    # Main output
                    # side_1_f = slim.convolution2d_transpose(side_2, 16, 2, 1, scope='score-multi1-up')
                    # side_1_f = crop_features(side_1_f, im_size)
                    # utils.collect_named_outputs(end_points_collection, 'osvos/side-multi1-cr', side_1_f)

                    side_2_f = slim.convolution2d_transpose(side_2, 16, 4, 2, scope='score-multi2-up')
                    side_2_f = crop_features(side_2_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/side-multi2-cr', side_2_f)
                    side_3_f = slim.convolution2d_transpose(side_3, 16, 8, 4, scope='score-multi3-up')
                    side_3_f = crop_features(side_3_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/side-multi3-cr', side_3_f)
                    side_4_f = slim.convolution2d_transpose(side_4, 16, 16, 8, scope='score-multi4-up')
                    side_4_f = crop_features(side_4_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/side-multi4-cr', side_4_f)
                    side_5_f = slim.convolution2d_transpose(side_5, 16, 32, 16, scope='score-multi5-up')
                    side_5_f = crop_features(side_5_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/side-multi5-cr', side_5_f)
                concat_side = tf.concat([side_2_f, side_3_f, side_4_f, side_5_f], axis=3)

                net = slim.conv2d(concat_side, 1, [1, 1], scope='upscore-fuse')

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points


def osvos_arg_scope(weight_decay=0.0002):
    """Defines the OSVOS arg scope.
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=None,
                        padding='SAME') as arg_sc:
        return arg_sc


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# Set deconvolutional layers to compute bilinear interpolation
def interp_surgery(variables):
    interp_tensors = []
    for v in variables:
        if '-up' in v.name:
            h, w, k, m = v.get_shape()
            tmp = np.zeros((m, k, h, w))
            if m != k:
                raise ValueError('input + output channels need to be the same')
            if h != w:
                raise ValueError('filters need to be square')
            up_filter = upsample_filt(int(h))
            tmp[range(m), range(k), :, :] = up_filter
            interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors


# Move preprocessing into Tensorflow
def preprocess_img(image):
    """Preprocess the image to adapt it to network requirements
    Image we want to input the network (W,H,3) numpy array
    Returns:
    Image ready to input the network (1,W,H,3)
    """
    if type(image) is not np.ndarray:
        image = np.array(Image.open(image), dtype=np.uint8)
    in_ = image[:, :, ::-1]
    in_ = np.subtract(in_, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    # in_ = tf.subtract(tf.cast(in_, tf.float32), np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    in_ = np.expand_dims(in_, axis=0)
    # in_ = tf.expand_dims(in_, 0)
    return in_


# Move preprocessing into Tensorflow
def preprocess_labels(label):
    """Preprocess the labels to adapt them to the loss computation requirements
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """
    if type(label) is not np.ndarray:
        label = np.array(Image.open(label).split()[0], dtype=np.uint8)
    max_mask = np.max(label) * 0.5
    label = np.greater(label, max_mask)
    label = np.expand_dims(np.expand_dims(label, axis=0), axis=3)
    # label = tf.cast(np.array(label), tf.float32)
    # max_mask = tf.multiply(tf.reduce_max(label), 0.5)
    # label = tf.cast(tf.greater(label, max_mask), tf.float32)
    # label = tf.expand_dims(tf.expand_dims(label, 0), 3)
    return label


def load_vgg_imagenet(ckpt_path):
    """Initialize the network parameters from the VGG-16 pre-trained model provided by TF-SLIM
    Path to the checkpoint
    Returns:
    Function that takes a session and initializes the network
    """
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_corresp = dict()
    for v in var_to_shape_map:
        if "conv" in v:
            #slim.get_model_variables(v.replace("vgg_16", "osvos"))[0]
            vars_corresp[v] = slim.get_model_variables(v.replace("vgg_16", "osvos"))[0]
    init_fn = slim.assign_from_checkpoint_fn(
        ckpt_path,
        vars_corresp)
    return init_fn


def class_balanced_cross_entropy_loss(output, label):
    """Define the class balanced cross entropy loss to train the network
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = tf.cast(tf.greater(label, 0.5), tf.float32)

    num_labels_pos = tf.reduce_sum(labels)
    num_labels_neg = tf.reduce_sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    return final_loss


def class_balanced_cross_entropy_loss_theoretical(output, label):
    """Theoretical version of the class balanced cross entropy loss to train the network (Produces unstable results)
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """
    output = tf.nn.sigmoid(output)

    labels_pos = tf.cast(tf.greater(label, 0), tf.float32)
    labels_neg = tf.cast(tf.less(label, 1), tf.float32)

    num_labels_pos = tf.reduce_sum(labels_pos)
    num_labels_neg = tf.reduce_sum(labels_neg)
    num_total = num_labels_pos + num_labels_neg

    loss_pos = tf.reduce_sum(tf.multiply(labels_pos, tf.log(output + 0.00001)))
    loss_neg = tf.reduce_sum(tf.multiply(labels_neg, tf.log(1 - output + 0.00001)))

    final_loss = -num_labels_neg / num_total * loss_pos - num_labels_pos / num_total * loss_neg

    return final_loss


def load_caffe_weights(weights_path):
    """Initialize the network parameters from a .npy caffe weights file
    Path to the .npy file containing the value of the network parameters
    Returns:
    Function that takes a session and initializes the network
    """
    osvos_weights = np.load(weights_path).item()
    vars_corresp = dict()
    vars_corresp['osvos/conv1/conv1_1/weights'] = osvos_weights['conv1_1_w']
    vars_corresp['osvos/conv1/conv1_1/biases'] = osvos_weights['conv1_1_b']
    vars_corresp['osvos/conv1/conv1_2/weights'] = osvos_weights['conv1_2_w']
    vars_corresp['osvos/conv1/conv1_2/biases'] = osvos_weights['conv1_2_b']

    vars_corresp['osvos/conv2/conv2_1/weights'] = osvos_weights['conv2_1_w']
    vars_corresp['osvos/conv2/conv2_1/biases'] = osvos_weights['conv2_1_b']
    vars_corresp['osvos/conv2/conv2_2/weights'] = osvos_weights['conv2_2_w']
    vars_corresp['osvos/conv2/conv2_2/biases'] = osvos_weights['conv2_2_b']

    vars_corresp['osvos/conv3/conv3_1/weights'] = osvos_weights['conv3_1_w']
    vars_corresp['osvos/conv3/conv3_1/biases'] = osvos_weights['conv3_1_b']
    vars_corresp['osvos/conv3/conv3_2/weights'] = osvos_weights['conv3_2_w']
    vars_corresp['osvos/conv3/conv3_2/biases'] = osvos_weights['conv3_2_b']
    vars_corresp['osvos/conv3/conv3_3/weights'] = osvos_weights['conv3_3_w']
    vars_corresp['osvos/conv3/conv3_3/biases'] = osvos_weights['conv3_3_b']

    vars_corresp['osvos/conv4/conv4_1/weights'] = osvos_weights['conv4_1_w']
    vars_corresp['osvos/conv4/conv4_1/biases'] = osvos_weights['conv4_1_b']
    vars_corresp['osvos/conv4/conv4_2/weights'] = osvos_weights['conv4_2_w']
    vars_corresp['osvos/conv4/conv4_2/biases'] = osvos_weights['conv4_2_b']
    vars_corresp['osvos/conv4/conv4_3/weights'] = osvos_weights['conv4_3_w']
    vars_corresp['osvos/conv4/conv4_3/biases'] = osvos_weights['conv4_3_b']

    vars_corresp['osvos/conv5/conv5_1/weights'] = osvos_weights['conv5_1_w']
    vars_corresp['osvos/conv5/conv5_1/biases'] = osvos_weights['conv5_1_b']
    vars_corresp['osvos/conv5/conv5_2/weights'] = osvos_weights['conv5_2_w']
    vars_corresp['osvos/conv5/conv5_2/biases'] = osvos_weights['conv5_2_b']
    vars_corresp['osvos/conv5/conv5_3/weights'] = osvos_weights['conv5_3_w']
    vars_corresp['osvos/conv5/conv5_3/biases'] = osvos_weights['conv5_3_b']

    vars_corresp['osvos/conv2_1_16/weights'] = osvos_weights['conv2_2_16_w']
    vars_corresp['osvos/conv2_1_16/biases'] = osvos_weights['conv2_2_16_b']
    vars_corresp['osvos/conv2_2_16/weights'] = osvos_weights['conv2_2_16_w']
    vars_corresp['osvos/conv2_2_16/biases'] = osvos_weights['conv2_2_16_b']

    vars_corresp['osvos/conv3_1_16/weights'] = osvos_weights['conv3_3_16_w']
    vars_corresp['osvos/conv3_1_16/biases'] = osvos_weights['conv3_3_16_b']
    vars_corresp['osvos/conv3_2_16/weights'] = osvos_weights['conv3_3_16_w']
    vars_corresp['osvos/conv3_2_16/biases'] = osvos_weights['conv3_3_16_b']
    vars_corresp['osvos/conv3_3_16/weights'] = osvos_weights['conv3_3_16_w']
    vars_corresp['osvos/conv3_3_16/biases'] = osvos_weights['conv3_3_16_b']

    vars_corresp['osvos/conv4_1_16/weights'] = osvos_weights['conv4_3_16_w']
    vars_corresp['osvos/conv4_1_16/biases'] = osvos_weights['conv4_3_16_b']
    vars_corresp['osvos/conv4_1_16/weights'] = osvos_weights['conv4_3_16_w']
    vars_corresp['osvos/conv4_1_16/biases'] = osvos_weights['conv4_3_16_b']
    vars_corresp['osvos/conv4_3_16/weights'] = osvos_weights['conv4_3_16_w']
    vars_corresp['osvos/conv4_3_16/biases'] = osvos_weights['conv4_3_16_b']

    vars_corresp['osvos/conv5_1_16/weights'] = osvos_weights['conv5_3_16_w']
    vars_corresp['osvos/conv5_1_16/biases'] = osvos_weights['conv5_3_16_b']
    vars_corresp['osvos/conv5_2_16/weights'] = osvos_weights['conv5_3_16_w']
    vars_corresp['osvos/conv5_2_16/biases'] = osvos_weights['conv5_3_16_b']
    vars_corresp['osvos/conv5_3_16/weights'] = osvos_weights['conv5_3_16_w']
    vars_corresp['osvos/conv5_3_16/biases'] = osvos_weights['conv5_3_16_b']

    vars_corresp['osvos/score-dsn_2/weights'] = osvos_weights['score-dsn_2_w']
    vars_corresp['osvos/score-dsn_2/biases'] = osvos_weights['score-dsn_2_b']
    vars_corresp['osvos/score-dsn_3/weights'] = osvos_weights['score-dsn_3_w']
    vars_corresp['osvos/score-dsn_3/biases'] = osvos_weights['score-dsn_3_b']
    vars_corresp['osvos/score-dsn_4/weights'] = osvos_weights['score-dsn_4_w']
    vars_corresp['osvos/score-dsn_4/biases'] = osvos_weights['score-dsn_4_b']
    vars_corresp['osvos/score-dsn_5/weights'] = osvos_weights['score-dsn_5_w']
    vars_corresp['osvos/score-dsn_5/biases'] = osvos_weights['score-dsn_5_b']

    vars_corresp['osvos/upscore-fuse/weights'] = osvos_weights['new-score-weighting_w']
    vars_corresp['osvos/upscore-fuse/biases'] = osvos_weights['new-score-weighting_b']
    return slim.assign_from_values_fn(vars_corresp)


def parameter_lr():
    """Specify the relative learning rate for every parameter. The final learning rate
    in every parameter will be the one defined here multiplied by the global one
    Returns:
    Dictionary with the relative learning rate for every parameter
    """

    vars_corresp = dict()
    vars_corresp['osvos/conv1/conv1_1/weights'] = 1
    vars_corresp['osvos/conv1/conv1_1/biases'] = 2
    vars_corresp['osvos/conv1/conv1_2/weights'] = 1
    vars_corresp['osvos/conv1/conv1_2/biases'] = 2

    vars_corresp['osvos/conv2/conv2_1/weights'] = 1
    vars_corresp['osvos/conv2/conv2_1/biases'] = 2
    vars_corresp['osvos/conv2/conv2_2/weights'] = 1
    vars_corresp['osvos/conv2/conv2_2/biases'] = 2

    vars_corresp['osvos/conv3/conv3_1/weights'] = 1
    vars_corresp['osvos/conv3/conv3_1/biases'] = 2
    vars_corresp['osvos/conv3/conv3_2/weights'] = 1
    vars_corresp['osvos/conv3/conv3_2/biases'] = 2
    vars_corresp['osvos/conv3/conv3_3/weights'] = 1
    vars_corresp['osvos/conv3/conv3_3/biases'] = 2

    vars_corresp['osvos/conv4/conv4_1/weights'] = 1
    vars_corresp['osvos/conv4/conv4_1/biases'] = 2
    vars_corresp['osvos/conv4/conv4_2/weights'] = 1
    vars_corresp['osvos/conv4/conv4_2/biases'] = 2
    vars_corresp['osvos/conv4/conv4_3/weights'] = 1
    vars_corresp['osvos/conv4/conv4_3/biases'] = 2

    vars_corresp['osvos/conv5/conv5_1/weights'] = 1
    vars_corresp['osvos/conv5/conv5_1/biases'] = 2
    vars_corresp['osvos/conv5/conv5_2/weights'] = 1
    vars_corresp['osvos/conv5/conv5_2/biases'] = 2
    vars_corresp['osvos/conv5/conv5_3/weights'] = 1
    vars_corresp['osvos/conv5/conv5_3/biases'] = 2

    vars_corresp['osvos/conv2_1_16/weights'] = 1
    vars_corresp['osvos/conv2_1_16/biases'] = 2
    vars_corresp['osvos/conv2_2_16/weights'] = 1
    vars_corresp['osvos/conv2_2_16/biases'] = 2

    vars_corresp['osvos/conv3_1_16/weights'] = 1
    vars_corresp['osvos/conv3_1_16/biases'] = 2
    vars_corresp['osvos/conv3_2_16/weights'] = 1
    vars_corresp['osvos/conv3_2_16/biases'] = 2
    vars_corresp['osvos/conv3_3_16/weights'] = 1
    vars_corresp['osvos/conv3_3_16/biases'] = 2

    vars_corresp['osvos/conv4_1_16/weights'] = 1
    vars_corresp['osvos/conv4_1_16/biases'] = 2
    vars_corresp['osvos/conv4_2_16/weights'] = 1
    vars_corresp['osvos/conv4_2_16/biases'] = 2
    vars_corresp['osvos/conv4_3_16/weights'] = 1
    vars_corresp['osvos/conv4_3_16/biases'] = 2

    vars_corresp['osvos/conv5_1_16/weights'] = 1
    vars_corresp['osvos/conv5_1_16/biases'] = 2
    vars_corresp['osvos/conv5_2_16/weights'] = 1
    vars_corresp['osvos/conv5_2_16/biases'] = 2
    vars_corresp['osvos/conv5_3_16/weights'] = 1
    vars_corresp['osvos/conv5_3_16/biases'] = 2

    vars_corresp['osvos/score-dsn_2/weights'] = 0.1
    vars_corresp['osvos/score-dsn_2/biases'] = 0.2
    vars_corresp['osvos/score-dsn_3/weights'] = 0.1
    vars_corresp['osvos/score-dsn_3/biases'] = 0.2
    vars_corresp['osvos/score-dsn_4/weights'] = 0.1
    vars_corresp['osvos/score-dsn_4/biases'] = 0.2
    vars_corresp['osvos/score-dsn_5/weights'] = 0.1
    vars_corresp['osvos/score-dsn_5/biases'] = 0.2

    vars_corresp['osvos/upscore-fuse/weights'] = 0.01
    vars_corresp['osvos/upscore-fuse/biases'] = 0.02
    return vars_corresp
