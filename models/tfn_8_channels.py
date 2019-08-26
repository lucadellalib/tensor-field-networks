# -*- coding: utf-8 -*-
"""
    TFN 8 channels definition.
"""

import os
import sys

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "tfn"))

import layers
import utils
from utils import FLOAT_TYPE

NUM_CLASSES = 10


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(FLOAT_TYPE, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(FLOAT_TYPE, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training):
    """ Classification TFN, input is BxNx3, output Bx10 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    # radial basis functions
    rbf_low = 0.0
    rbf_high = 3.5
    rbf_count = 4
    rbf_spacing = (rbf_high - rbf_low) / rbf_count
    centers = tf.cast(tf.lin_space(rbf_low, rbf_high, rbf_count), FLOAT_TYPE)
    
    # rij : [None, N, N, 3]
    rij = utils.difference_matrix(point_cloud)

    # dij : [None, N, N]
    dij = utils.distance_matrix(point_cloud)

    # rbf : [None, N, N, rbf_count]
    gamma = 1. / rbf_spacing
    rbf = tf.exp(-gamma * tf.square(tf.expand_dims(dij, axis=-1) - centers))

    # channels
    layer_dims = [1, 8, 8, 8]
    num_layers = len(layer_dims) - 1

    # embed : [None, N, layer1_dim, 1]
    with tf.variable_scope(None, "embed"):
        embed = layers.self_interaction_layer_without_biases(tf.ones(shape=(tf.shape(point_cloud)[0], num_point, 1, 1)), layer_dims[0])

    input_tensor_list = {0: [embed]} 

    for layer, layer_dim in enumerate(layer_dims[1:]):
        with tf.variable_scope(None, "layer" + str(layer), values=[input_tensor_list]):
            input_tensor_list = layers.convolution(input_tensor_list, rbf, rij)
            input_tensor_list = layers.concatenation(input_tensor_list)
            input_tensor_list = layers.self_interaction(input_tensor_list, layer_dim)
            input_tensor_list = layers.nonlinearity(input_tensor_list)

    tfn_scalars = input_tensor_list[0][0]
    tfn_output_shape = tfn_scalars.get_shape().as_list()
    axis_to_squeeze = [i for i, e in enumerate(tfn_output_shape) if e == 1]
    tfn_output = tf.reduce_mean(tf.squeeze(tfn_scalars, axis=axis_to_squeeze), axis=1)
    fully_connected_layer = tf.get_variable("fully_connected_weights",
                                            [tfn_output_shape[-2], NUM_CLASSES], dtype=FLOAT_TYPE)
    output_biases = tf.get_variable("output_biases", [NUM_CLASSES], dtype=FLOAT_TYPE)

    # output : [None, NUM_CLASSES]
    output = tf.einsum("xy,hx->hy", fully_connected_layer, tfn_output) + output_biases

    return output, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # truth : [NUM_CLASSES]
    truth = tf.one_hot(tf.to_int32(label), NUM_CLASSES)
    # loss : [None]
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=truth, logits=pred)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar("classify loss", classify_loss)
    tf.add_to_collection("losses", classify_loss)
    return classify_loss


if __name__=="__main__":
	with tf.Graph().as_default():
		inputs = tf.zeros((32, 1024, 3))
		net = get_model(inputs, tf.constant(True))
		print(net)
