import tensorflow as tf
import time
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784], name="X")        # [batch_size, num_pixel]
X_img = tf.reshape(X, [-1, 28, 28, 1])                       # [batch_size, 가로, 세로, channel] img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10], name="Y")         # [batch_size, num_label]
keep_prob = tf.placeholder(tf.float32, name="keep_prob")     # probalbility for dropout

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))  # [filter_size, filter_size, channel, num_filter]
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  # strides: [batch, 가로, 세로, channel] // VALID
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # tf.nn.avg_pool  # padding same: assume strides=1
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)  # (?, 7, 7, 64)
#print(tf.shape(L2))
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])  # 7*7*64 = 448 * 7 = 3136

#####################################
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 300], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([300]))
L3 = tf.nn.relu(tf.matmul(L2_flat, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
#####################################

W4 = tf.get_variable("W4", shape=[300, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.xw_plus_b(L3, W4, b, name="hypothesis")
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

        if i % 20 == 0:  ############
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
print('learnig rate:',  learning_rate, 'training_epochs:', training_epochs, 'batch_size:', batch_size)

#돌리다가 오류가 났음
#Epoch: 0002 Iter: 0120 training cost = 0.025967613
#240 step Validation Accuracy: 0.9804
#까지 돌리고

#tensorflow.python.framework.errors_impl.UnknownError: Failed to rename: D:\OneDrive - Sejong University\텐서플로\aischool\lecture_4\deep_learning\runs\01-26-21-52-13\checkpoints\model-2.meta.tmpfcc8a70553804fe29e7c53e97da3f0cd to: D:\OneDrive - Sejong University\텐서플로\aischool\lecture_4\deep_learning\runs\01-26-21-52-13\checkpoints\model-2.meta : \udcbe׼\udcbc\udcbd\udcba\udcb0\udca1 \udcb0źεǾ\udcfa\udcbd\udcc0\udcb4ϴ\udcd9.
#; Input/output error

#Process finished with exit code 1

#다시 한번 돌려보니 에러가 안 나고 돌아가고 있음