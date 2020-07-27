import tensorflow as tf
import numpy as np


class DCGAN(object):

    def __init__(self, config):
        self.Z_dim = config["Z_dim"]
        self.dc_dim = config["dc_dim"]
        self.gc_dim = config["gc_dim"]
        self.c_dim = config["c_dim"]
        self.dataset = config["dataset"]
        self.image_size = config["image_size"]
        self.leakiness = config["relu_leakiness"]

        # Placeholders for input
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, self.c_dim])
        self.input_z = tf.placeholder(tf.float32, shape=[None, self.Z_dim])
        self.batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.G = self.generator(self.input_z)
        self.G_sample = self.generator(self.input_z, False)
        D_real, D_logit_real = self.discriminator(self.input_x, reuse=False)
        D_fake, D_logit_fake = self.discriminator(self.G, reuse=True)

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        t_vars = tf.trainable_variables()

        self.theta_D = [var for var in t_vars if 'd_' in var.name]
        self.theta_G = [var for var in t_vars if 'g_' in var.name]

    def sample_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def generator(self, z, train=True):
        with tf.variable_scope("generator") as scope:
            if not train:
                scope.reuse_variables()
            if self.dataset == 'bedroom':
                # project `z` and reshape
                self.z_ = self._fully_connected(z, self.gc_dim * 8 * int(self.image_size/16) * int(self.image_size/16), 'g_h0_lin')

                h0 = tf.reshape(self.z_, [-1, int(self.image_size/16), int(self.image_size/16), self.gc_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0))

                h1 = self._deconv(h0, [self.batch_size, int(self.image_size/8), int(self.image_size/8), self.gc_dim * 4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1))

                h2 = self._deconv(h1, [self.batch_size, int(self.image_size/4), int(self.image_size/4), self.gc_dim * 2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3 = self._deconv(h2, [self.batch_size, int(self.image_size/2), int(self.image_size/2), self.gc_dim * 1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4 = self._deconv(h3, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)

            elif self.dataset == 'mnist':
                self.z_ = self._fully_connected(z, self.gc_dim * 2 * int(self.image_size/4) * int(self.image_size/4), 'g_h0_lin')

                h0 = tf.reshape(self.z_, [-1, int(self.image_size/4), int(self.image_size/4), self.gc_dim * 2])
                h0 = tf.nn.relu(self.g_bn0(h0))

                h1 = self._deconv(h0, [self.batch_size, int(self.image_size/2), int(self.image_size/2), self.gc_dim * 1], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1))

                h2 = self._deconv(h1, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='g_h2')
                return tf.nn.tanh(h2)

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            if self.dataset == 'bedroom':
                h0 = self._relu(self._conv(x, self.dc_dim, name='d_h0_conv'), self.leakiness)
                h1 = self._relu(self.d_bn1(self._conv(h0, self.dc_dim * 2, name='d_h1_conv')), self.leakiness)
                h2 = self._relu(self.d_bn2(self._conv(h1, self.dc_dim * 4, name='d_h2_conv')), self.leakiness)
                h3 = self._relu(self.d_bn3(self._conv(h2, self.dc_dim * 8, name='d_h3_conv')), self.leakiness)
                D_logit = self._fully_connected(h3, 1, 'd_h4_lin')
            elif self.dataset == 'mnist':
                h0 = self._relu(self._conv(x, self.dc_dim, name='d_h0_conv'), self.leakiness)
                h1 = self._relu(self.d_bn1(self._conv(h0, self.dc_dim * 2, name='d_h1_conv')), self.leakiness)
                D_logit = self._fully_connected(h1, 1, 'd_h2_lin')

            D_prob = tf.nn.sigmoid(D_logit)
            return D_prob, D_logit

    def _relu(self, x, leakiness=0.2):
        return tf.nn.leaky_relu(x, leakiness)

    def _conv(self, x, out_filters, name):
        with tf.variable_scope(name):
            n = 5 * 5 * out_filters # he 초기화를 위한 크기 계산
            # filter size 가로, 세로, input channel, output channel 크기로 filter 초기화
            kernel = tf.get_variable(
                'W', [5, 5, x.get_shape()[-1], out_filters],
                tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, [1, 2, 2, 1], padding='SAME') # convolution 연산

    def _deconv(self, x, output_shape, name):
        with tf.variable_scope(name):
            kernel = tf.get_variable('W', [5, 5, output_shape[-1], x.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=0.1))
            deconv = tf.nn.conv2d_transpose(x, kernel, output_shape=output_shape, strides=[1, 2, 2, 1])

            return deconv

    def _fully_connected(self, x, out_dim, name):
        with tf.variable_scope(name):
            dim = tf.reduce_prod(x.get_shape()[1:]).eval()
            x = tf.reshape(x, [-1, dim])
            w = tf.get_variable(
                'DW', [dim, out_dim],
                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable('biases', [out_dim],
                                initializer=tf.constant_initializer())
            return tf.nn.xw_plus_b(x, w, b)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                      scale=True, is_training=train, scope=self.name, reuse=tf.AUTO_REUSE)



