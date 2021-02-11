import tensorflow as tf
import random
import time
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)
#one_hot = True, 4-> 0, 0, 0, 0, 1, 0, 0, 0, 0, 0

X = tf.placeholder(tf.float32, [None, 784], name="X")
Y = tf.placeholder(tf.float32, [None, 10], name="Y")
#dropout
keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # 살릴확률 0.5

#W1 = tf.Variable(tf.random_normal([784, 300]))
W1 = tf.get_variable("W1", shape=[784, 300], initializer=tf.contrib.layers.xavier_initializer()) #W1을 초기배정할 때 xavier로 써본 것
#1번째 레이어
b1 = tf.Variable(tf.random_normal([300]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
#2번째 레이어
W2 = tf.get_variable("W2", shape=[300, 150], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([150]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
#
W3 = tf.get_variable("W3", shape=[150, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))

hypothesis = tf.nn.xw_plus_b(L2, W3, b3, name="hypothesis")  # L3*W4 + b4 위의 matmul 하고 + 하는 것을 한번에 하는 코딩
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))  # Y = [0,0,0,1,0,0,0,0,0,0]
# [0.11, 0.01, 0.002, ... 1.9]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # True -> 1.0 False -> 0.0
summary_op = tf.summary.scalar("accuracy", accuracy)

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

learning_rate = 0.001
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # back propagation

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 500
batch_size = 100

# ======================================================================== 패스
timestamp = str(time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())))  # runs/1578546654/checkpoints/
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
val_summary_dir = os.path.join(out_dir, "summaries", "dev")
val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
# ========================================================================

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

max = 0  # max validation accuracy
dropout = 0.85  # prob to live
early_stopped = 0  # early_stopped time
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)  # iteration 55000/100 = 550

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # (100, 784), (100, 10)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: dropout}
        c, _, a = sess.run([cost, optimizer, summary_op], feed_dict=feed_dict)
        avg_cost += c / total_batch

    #......
    print('Epoch:', '%04d' % (epoch + 1), 'training cost =', '{:.9f}'.format(avg_cost))
    # ========================================================================텐서보드
    train_summary_writer.add_summary(a, epoch)
    val_accuracy, summaries = sess.run([accuracy, summary_op],
                                       feed_dict={X: mnist.validation.images,
                                                  Y: mnist.validation.labels,
                                                  keep_prob: 1.0})

    val_summary_writer.add_summary(summaries, epoch)
    # ========================================================================
    print('Validation Accuracy:', val_accuracy)
    if val_accuracy > max:
        max = val_accuracy
        early_stopped = epoch + 1
        saver.save(sess, checkpoint_prefix, global_step=early_stopped)

#......
print('Learning Finished!', '-'*30)
print('Validation Max Accuracy:', max)
print('Early stopped time:', early_stopped)

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print('Latest Model Test Accuracy:', test_accuracy)
print('learnig rate:',  learning_rate, 'training_epochs:', training_epochs, 'batch_size:', batch_size, 'drop out:', dropout)

# #중간에 5에서 6 epoch로 넘어갈 때
# WARNING:tensorflow:From C:\Users\chanh\anaconda3\envs\aischool\lib\site-packages\tensorflow\python\training\saver.py:966: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use standard file APIs to delete files with this prefix. 라고 오류가 뜸