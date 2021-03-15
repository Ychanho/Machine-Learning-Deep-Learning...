import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
# ResNet 모델의 구조만 정의
class ResNet:
    def __init__(self, config):

        self._num_residual_units = config.num_residual_units #unit 개수, 즉 n의 크기
        self._batch_size = config.batch_size
        self._relu_leakiness = config.relu_leakiness
        self._num_classes = config.num_classes # label 개수 (10개-airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        self._l2_reg_lambda = config.l2_reg_lambda

        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="X")
        self.Y = tf.placeholder(tf.float32, [None, self._num_classes], name="Y")
        self.extra_train_ops = [] # batch norm 파라미터 학습을 위한 operation 저장할 변수

        filters = [16, 16, 32, 64] # 층별 filter 개수 (output channel)
        activate_before_residual = [True, False, False] # activation 연산 수행 시점의 residual 연산 전후 여부

        with tf.variable_scope('init'): # 최초 convolutional layer
          x = self._conv('init_conv', self.X, 3, 3, filters[0], strides=[1, 1, 1, 1])

        with tf.variable_scope('unit_1_0'):
          x = self._residual(x, filters[0], filters[1], activate_before_residual[0], strides=[1, 1, 1, 1]) #첫 번째 residual unit
        for i in range(1, self._num_residual_units): #나머지 residual unit (n이 3이라면 두번째, 세번째 unit들)
          with tf.variable_scope('unit_1_%d' % i):
            x = self._residual(x, filters[1], filters[1], strides=[1, 1, 1, 1])

        with tf.variable_scope('unit_2_0'):
          x = self._residual(x, filters[1], filters[2], activate_before_residual[1], strides=[1, 2, 2, 1])
        for i in range(1, self._num_residual_units):
          with tf.variable_scope('unit_2_%d' % i):
            x = self._residual(x, filters[2], filters[2], strides=[1, 1, 1, 1])

        with tf.variable_scope('unit_3_0'):
          x = self._residual(x, filters[2], filters[3], activate_before_residual[2], strides=[1, 2, 2, 1])
        for i in range(1, self._num_residual_units):
          with tf.variable_scope('unit_3_%d' % i):
            x = self._residual(x, filters[3], filters[3], strides=[1, 1, 1, 1])

        with tf.variable_scope('unit_last'):
          x = self._batch_norm('final_bn', x)
          x = self._relu(x, self._relu_leakiness)
          x = self._global_avg_pool(x) # (?, 8, 8, 64) 크기의 feature maps을 average pooling을 통해 (?, 8, 8)로 요약 및 축소

        with tf.variable_scope('logit'):
          logits = self._fully_connected(x, self._num_classes)
          self.predictions = tf.nn.softmax(logits)
          self.predictions = tf.argmax(self.predictions, 1, name="predictions")

        with tf.variable_scope('loss'):
          xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y)
          self.loss = tf.reduce_mean(xent, name='xent')
          self.loss += self._decay()

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters # he 초기화를 위한 크기 계산
            # filter size 가로, 세로, input channel, output channel 크기로 filter 초기화
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME') # convolution 연산

    def _residual(self, x, in_filter, out_filter, activate_before_residual=False, strides=[1, 1, 1, 1]):

        if activate_before_residual:
            with tf.variable_scope('common_activation'):
                x = self._batch_norm('init_bn', x)          # batch마다 표준 정규화 같이 정규화 해주는 것
                x = self._relu(x, self._relu_leakiness)
                orig_x = x # skip connection 때 넘 겨줄 feature orig_x에 복사
        else:
            with tf.variable_scope('residual_activation'):
                orig_x = x # skip connection 때 넘 겨줄 feature orig_x에 복사
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self._relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, strides)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self._relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter: # stride 크기가 2일 때 channel 크기가 안맞는 경우 크기 조정을 통해 skip connection이 원활하게 조정
                orig_x = tf.nn.avg_pool(orig_x, strides, strides, 'VALID')
                orig_x = tf.pad(    # [ batch, 가로, 세로, 채널수]
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]]) # 부족한 채널수의 절반만큼 앞뒤로 pad해주는 것
            x += orig_x # skip connection

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _batch_norm(self, name, x):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)

            self.extra_train_ops.append(moving_averages.assign_moving_average(
                moving_mean, mean, 0.9))
            self.extra_train_ops.append(moving_averages.assign_moving_average(
                moving_variance, variance, 0.9))

            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _fully_connected(self, x, out_dim):
        # [bach_size, x=1, y=1, channel] 1,1인 이유는 avg pooling을 해서 1이 된것
        dim = tf.reduce_prod(x.get_shape()[1:]).eval()  # 안에 있는 값들을 다 곱해서 1개로 줄이는 것
        # tf.reduce_mean(x) 은 다 더하고 평균 내는 것
        print(dim)
        x = tf.reshape(x, [-1, dim])    # x = [1, 10, 50] dim=5  이라면 -1을 써 놨기 때문에 자동으로 500/50을 계산한 값이 들어감
        w = tf.get_variable(
            'DW', [dim, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        # (?, 8, 8, 64) 크기의 feature maps을 average pooling을 통해 (?, 8, 8)로 요약 및 축소
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables(): # 학습 가능한 모든 변수들의
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var)) # l2 norm을 구해서 costs에 저장

        return tf.multiply(self._l2_reg_lambda, tf.add_n(costs)) # costs안에 값들을 모두 더한 뒤 lambda 값에 곱해줌