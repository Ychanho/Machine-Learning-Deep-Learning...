import tensorflow as tf
import time
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784], name="X")
Y = tf.placeholder(tf.float32, [None, 10], name="Y")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # 살릴확률 0.5~0.8

# W1 = tf.Variable(tf.random_normal([784, 300]))
W1 = tf.get_variable("W1", shape=[784, 300], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([300]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[300, 200], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([200]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[200, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))

hypothesis = tf.nn.xw_plus_b(L2, W3, b3, name="hypothesis")  # L3*W4 + b4
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
summary_op = tf.summary.scalar("loss", cost)

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 50
batch_size = 100

timestamp = str(time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())))  # runs/1578546654/checkpoints/
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
val_summary_dir = os.path.join(out_dir, "summaries", "valid")
val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

max = 0
early_stopped = 0
global_step = 0
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    print(f"total_batch: {total_batch}")

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.8}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

        if i % 20 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'Iter:', '%04d' % i, 'training cost =', '{:.9f}'.format(avg_cost))
            val_accuracy, summaries = sess.run([accuracy, summary_op],
                                               feed_dict={X: mnist.validation.images, Y: mnist.validation.labels,
                                                          keep_prob: 1.0})
            # val_summary_writer.add_summary(summaries, epoch)
            val_summary_writer.add_summary(summaries, global_step)
            print(f'{(epoch+1) * i} step Validation Accuracy:', val_accuracy)
            if val_accuracy > max:
                max = val_accuracy
                early_stopped = epoch + 1
                saver.save(sess, checkpoint_prefix, global_step=early_stopped)

        global_step += batch_size

print('Learning Finished!')
print('Validation Max Accuracy:', max)
print('Early stopped time:', early_stopped)

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print('Test Latest Accuracy:', test_accuracy)