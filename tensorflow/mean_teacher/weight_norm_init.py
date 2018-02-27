"""
Layers with weight normalization and mean-only batch normalization but only during initialization. 

See https://arxiv.org/abs/1602.07868 (Salimans & Kingma, 2016) 

The code is adapted from
https://github.com/openai/pixel-cnn/blob/fc86dbce1d508fa79f8e9a7d1942d229249a5366/pixel_cnn_pp/nn.py
"""

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope


@add_arg_scope
def fully_connected(inputs, num_outputs,
                    activation_fn=None, init_scale=1., init=False,
                    eval_mean_ema_decay=0.999, is_training=None, scope=None, layer_collection=None, reuse=None):
    #pylint: disable=invalid-name
    with tf.variable_scope(scope, "fully_connected"):
        if is_training is None:
            is_training = tf.constant(True)
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V',
                                [int(inputs.get_shape()[1]), num_outputs],
                                tf.float32,
                                tf.random_normal_initializer(0, 0.05),
                                trainable=True)
            b = tf.get_variable('b', shape=[num_outputs], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=True)
            with tf.control_dependencies([V.assign(tf.nn.l2_normalize(V.initialized_value(), [0]))]):
                x_init = tf.matmul(inputs, V)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([V.assign(V*scale_init),b.assign_add(-m_init * scale_init)]):
                x_init = (tf.reshape(scale_init, [1, num_outputs]) *
                      (x_init - tf.reshape(m_init, [1, num_outputs])))
            if activation_fn is not None:
                x_init = activation_fn(x_init)
            return x_init
        else:
            V, b = [tf.get_variable(var_name) for var_name in ['V', 'b']]

            preactivations = tf.matmul(inputs, V) + tf.reshape(b, [1, num_outputs])

            # apply nonlinearity
            if activation_fn is not None:
                activations = activation_fn(inputs)
            else:
                activations = preactivations

            if layer_collection is not None:
              #layer_collection.register_fully_connected((V,b), inputs, preactivations, reuse=reuse) 
              layer_collection.register_fully_connected(V, inputs, preactivations, reuse=reuse) 

            return activations


@add_arg_scope
def conv2d(inputs, num_outputs,
           kernel_size=[3, 3], stride=[1, 1], padding='SAME',
           activation_fn=None, init_scale=1., init=False,
           eval_mean_ema_decay=0.999, is_training=None, scope=None, layer_collection=None, reuse=None):
    #pylint: disable=invalid-name
    with tf.variable_scope(scope, "conv2d"):
        if is_training is None:
            is_training = tf.constant(True)
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', kernel_size + [int(inputs.get_shape()[-1]), num_outputs],
                                tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
            b = tf.get_variable('b', shape=[num_outputs], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=True)
            with tf.control_dependencies([V.assign(tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2]))]):
                x_init = tf.nn.conv2d(inputs, V, [1] + stride + [1], padding)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            with tf.control_dependencies([V.assign(V*scale_init),b.assign_add(-m_init * scale_init)]):
                x_init = (tf.reshape(scale_init, [1, 1, 1, num_outputs]) *
                      (x_init - tf.reshape(m_init, [1, 1, 1, num_outputs])))

            if activation_fn is not None:
                x_init = activation_fn(x_init)
            return x_init

        else:
            V, b = [tf.get_variable(var_name) for var_name in ['V', 'b']]

            # calculate convolutional layer output
            preactivations = tf.nn.conv2d(inputs, V, [1] + stride + [1], padding)
            preactivations = tf.nn.bias_add(preactivations, b)

            # apply nonlinearity
            if activation_fn is not None:
                activations = activation_fn(preactivations)
            else:
                activations = activations

            if layer_collection is not None:
              #layer_collection.register_conv2d((V,b), tuple([1] + stride + [1]), padding, inputs, preactivations, reuse=reuse) 
              layer_collection.register_conv2d(V, tuple([1] + stride + [1]), padding, inputs, preactivations, reuse=reuse) 

            return activations
