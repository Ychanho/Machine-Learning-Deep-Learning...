#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
from python_basic import data_helpers as dh
import smart_open
import pickle
from python_basic.text_rnn import TextRNN

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("x_test_file", "./data/test/x_agnews_test.txt", "Data source for the ODP training")
tf.flags.DEFINE_string("t_test_file", "./data/test/t_agnews_test.txt", "Data source for the ODP training")

# Eval Parameters
tf.flags.DEFINE_string("dir", "./runs/1585414264", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS

x_raw, y_test, length_test = dh.load_data(FLAGS.x_test_file, FLAGS.t_test_file)
y_test = np.argmax(y_test, axis=1)

with smart_open.smart_open(os.path.join(FLAGS.dir, "vocab"), 'rb') as f:
    word_id_dict = pickle.load(f)
with smart_open.smart_open(os.path.join(FLAGS.dir, "config"), 'rb') as f:
    config = pickle.load(f)

x_test = dh.text_to_index(x_raw, word_id_dict, 0)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.dir, "checkpoints"))
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        rnn = TextRNN(config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, checkpoint_file)

        # Generate batches for one epoch
        batches = dh.batch_iter(list(zip(x_test, length_test)), config["batch_size"], 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for batch in batches:
            x_test_batch, length_batch = zip(*batch)
            x_test_batch = dh.batch_tensor(x_test_batch)
            batch_predictions = sess.run(rnn.predictions, {rnn.input_x: x_test_batch, rnn.sequence_length: length_batch, rnn.batch_size: len(batch), rnn.dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    print(len(all_predictions))
    print(len(y_test))
    print(sum(all_predictions == y_test))
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
