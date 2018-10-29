from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4


def generator_drnet(image, options, reuse=False, name='generator_styled'):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, dilation_rate=2, name='res'):
            p = int(((ks - 1) / 2)*dilation_rate)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d_dilation(y, dim, ks, s, dilation_rate=dilation_rate, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d_dilation(y, dim, ks, s, dilation_rate=dilation_rate, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim*2, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))

        r1 = residule_block(c1, options.gf_dim*2, dilation_rate=1,name='g_r1')
        r2 = residule_block(r1, options.gf_dim*2, dilation_rate=2, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*2, dilation_rate=4, name='g_r3')

        c3 = tf.nn.relu(instance_norm(conv2d_dilation(r3, options.gf_dim*2, 3, 1, dilation_rate=2, padding='VALID', name='g_e3_c'), 'g_e3_bn'))
        c3 = tf.pad(c3, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        delta = tf.nn.tanh(conv2d(c3, options.output_c_dim, 3, 1, padding='VALID', name='g_pred_c'))
        #return tf.clip_by_value(delta + image, -1.0, 1.0)
        return delta + image, delta


def generator_drnet_styled(image, image_new, options, reuse=False, name='generator_styled'):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, dilation_rate=2, name='res'):
            p = int(((ks - 1) / 2)*dilation_rate)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d_dilation(y, dim, ks, s, dilation_rate=dilation_rate, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d_dilation(y, dim, ks, s, dilation_rate=dilation_rate, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        c01 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c02 = tf.pad(image_new, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c11 = tf.nn.relu(instance_norm(conv2d(c01, options.gf_dim, 7, 1, padding='VALID', name='g_e01_c'), 'g_e01_bn'))
        c12 = tf.nn.relu(instance_norm(conv2d(c02, options.gf_dim, 7, 1, padding='VALID', name='g_e02_c'), 'g_e02_bn'))

        r1 = residule_block(tf.concat([c11, c12], -1), options.gf_dim*2, dilation_rate=1,name='g_r1')
        r2 = residule_block(r1, options.gf_dim*2, dilation_rate=2, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*2, dilation_rate=4, name='g_r3')

        c3 = tf.nn.relu(instance_norm(conv2d_dilation(r3, options.gf_dim*2, 3, 1, dilation_rate=2, padding='VALID', name='g_e3_c'), 'g_e3_bn'))
        c3 = tf.pad(c3, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        delta = tf.nn.tanh(conv2d(c3, options.output_c_dim, 3, 1, padding='VALID', name='g_pred_c'))
        #return tf.clip_by_value(delta + image, -1.0, 1.0), delta
        return delta + image, delta


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
