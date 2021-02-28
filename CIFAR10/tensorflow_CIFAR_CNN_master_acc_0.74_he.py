import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.keras.datasets.cifar10 import load_data

X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="X")
Y = tf.placeholder(tf.float32, [None, 10], name="Y")  # 0 0 0 0 1 0 0 0 0 0 one-hot encoding
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# 32*32*3
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01)) # fillter 32
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
# 16*16*32
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # fillter 64
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
# 8*8*64
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)) # fillter 128
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
# (?,4,4,128)
L3_flat = tf.reshape(L3, [-1, 4 * 4 * 128])

##########################################################
# Add fully connected layer
W4 = tf.get_variable("W4", shape=[4 * 4 * 128, 2 * 128], initializer=tf.initializers.he_normal())
b4 = tf.Variable(tf.random_normal([2*128]))
FC1 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)

W5 = tf.get_variable("W5", shape=[2 * 128, 64], initializer=tf.initializers.he_normal())
b5 = tf.Variable(tf.random_normal([64]))
FC2 = tf.nn.relu(tf.matmul(FC1, W5) + b5)
##########################################################

W6 = tf.get_variable("W6", shape=[64, 10], initializer=tf.initializers.he_normal())
b6 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.xw_plus_b(FC2, W6, b6, name="hypothesis")
#(?, 10)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
summary_op = tf.summary.scalar("loss", cost)

learning_rate = 0.001
"""learning_rate 설정"""
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
"""Optimizer 설정"""
# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 100
batch_size = 512
dropout_keep = 0.6

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
val_summary_dir = os.path.join(out_dir, "summaries", "dev")
val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

max = 0
early_stopped = 0
(x_train_val, y_train_val), (x_test, y_test) = load_data()

shuffle_indices = np.random.permutation(np.arange(len(y_train_val)))  # [0,1,2,3,4...len] -> [4509,11,1356,4...len]
shuffled_x = np.asarray(x_train_val[shuffle_indices])
shuffled_y = y_train_val[shuffle_indices]

val_sample_index = -1 * int(0.1 * float(len(y_train_val)))  # -5000
x_train, x_val = shuffled_x[:val_sample_index], shuffled_x[val_sample_index:]
y_train, y_val = shuffled_y[:val_sample_index], shuffled_y[val_sample_index:]

x_test = np.asarray(x_test)
y_train_one_hot = np.eye(10)[y_train]  # [9, 8] -> [[0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0]
print(np.shape(y_train_one_hot))  # (45000, 1, 10)
y_train_one_hot = np.squeeze(y_train_one_hot, axis=1)  # (45000, 10)
print(np.shape(y_train_one_hot))
y_test_one_hot = np.eye(10)[y_test]
y_test_one_hot = np.squeeze(y_test_one_hot, axis=1)
y_val_one_hot = np.eye(10)[y_val]
y_val_one_hot = np.squeeze(y_val_one_hot, axis=1)

def next_batch(batch_size, data):
    data = np.array(data)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    shuffled_data = data[shuffle_indices]
    # num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    num_batches_per_epoch = int(len(data)/batch_size)

    for batch_num in range(num_batches_per_epoch):  # 450 - 0...449
        start_index = batch_num * batch_size  # 0, 100, 200, ..., 44900
        end_index = min((batch_num + 1) * batch_size, len(data))  # 100, 200, 300, ..., 44954
        yield shuffled_data[start_index:end_index]

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(y_train) / batch_size)  # 45000 / 100 = 450
    batches = next_batch(batch_size, list(zip(x_train, y_train_one_hot)))  # [(image, label), (image, label), ...]
    for batch in batches:
        batch_xs, batch_ys = zip(*batch)  # unzip  (image, label) -> image, label
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: dropout_keep}
        c, _, s = sess.run([cost, optimizer, summary_op], feed_dict=feed_dict)
        avg_cost += c / total_batch
    train_summary_writer.add_summary(s, epoch)

    print('Epoch:', '%04d' % (epoch + 1), 'training cost =', '{:.9f}'.format(avg_cost))
    val_accuracy, summaries = sess.run([accuracy, summary_op],
                                       feed_dict={X: x_val, Y: y_val_one_hot,
                                                  keep_prob: 1.0})
    val_summary_writer.add_summary(summaries, epoch)
    print('Validation Accuracy:', val_accuracy)
    if val_accuracy > max:
        max = val_accuracy
        early_stopped = epoch + 1
        saver.save(sess, checkpoint_prefix, global_step=early_stopped)

print('Learning Finished!', ('-'*50))
print('Validation Max Accuracy:', max)
print('Early stopped time:', early_stopped)

test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test_one_hot, keep_prob: 1.0})
print('Test Latest Accuracy:', test_accuracy)


print('Dropout:', dropout_keep, 'learnig rate:',  learning_rate, 'training_epochs:', training_epochs, 'batch_size:', batch_size, 'Test Latest Accuracy:', test_accuracy)
print(dropout_keep, learning_rate, training_epochs, batch_size, test_accuracy)