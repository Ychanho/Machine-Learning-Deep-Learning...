import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# model 정의
X = tf.placeholder(tf.float32, [None, 784])  # (batch, 784)
Y = tf.placeholder(tf.float32, [None, 10])  # (batch, 10)  [0 0 0 0 1 0 0 0 0 0]
#첫번째 레이어
W1 = tf.Variable(tf.random_normal([784, 350]))
b1 = tf.Variable(tf.random_normal([350]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)  # (batch, 784) x (784, 300) -> (batch, 300)
#2번째 레이어
W2 = tf.Variable(tf.random_normal([350, 250]))
b2 = tf.Variable(tf.random_normal([250]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)  # (batch, 300) x (300, 200) -> (batch, 200)
#output 레이어
W3 = tf.Variable(tf.random_normal([250, 10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3  # 0~9        # (batch, 200) x (200, 10) -> (batch, 10)

# define cost/loss & optimizer

# hypothesis = tf.nn.softmax(hypothesis) # [0.6784 0.137  0127, ...] -> [0.09  0.01  0.01  0.6]
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(  # (batch, 1) -> (1, ) (dimension)
    logits=hypothesis, labels=Y))

learning_rate = 0.003 #경험적으로 수정
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # back prop automatically

#------------------------------------------------모델생성 완료

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 30
batch_size = 100

# train model
max = 0.0
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)  # 55000/100 = 550
    for i in range(total_batch):  # iteration
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'training cost =', '{:.9f}'.format(avg_cost))
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))  # [0 0 0 0 1 0 0 0 0 0]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # True -> 1.0 False -> 0.0

    test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print('Test Accuracy:', test_accuracy)
    if test_accuracy > max:
        max = test_accuracy

print('=' * 100)
print('Learning Finished!')
print('Test Max Accuracy:', max)
print('learnig rate:',  learning_rate, 'training_epochs:', training_epochs,)
# 결과 0.9698
