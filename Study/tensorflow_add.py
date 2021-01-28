import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) #also tf.float32 implicily
node3 = tf.add(node1, node2)

print("node1:", node1, "node2:", node2)
print("node3: ",node3)

sess = tf.Session()
print("sess.run(node1, node2):", sess.run([node1, node2]))
print("sess.run(node3):", sess.run(node3))
##############################################################3

# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# c = tf.placeholder(tf.float32)
#
# adder_node = a+b+c
#
# sess = tf.Session()
# print(sess.run(adder_node, feed_dict={a:3, b:4.5, c:2.5}))
# print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4],c:[1,5]}))
# ####################################################################
#
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# y = tf.add(a, b)

