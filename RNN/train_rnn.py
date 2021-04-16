#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import re
import smart_open
import pickle
# import data_helpers as dh

from tensorflow import data_helpers as dh
from tensorflow.text_rnn import TextRNN

# from text_rnn import TextRNN
from gensim.models.keyedvectors import KeyedVectors

# Parameters
tf.flags.DEFINE_float("val_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("x_train_file", "./data/train/x_agnewsTrain.txt", "Data source for the training")  # world, sports, business, sci/tech
tf.flags.DEFINE_string("t_train_file", "./data/train/t_agnewsTrain.txt", "Data source for the training")
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings (default: None)")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of word embedding (default: 128)")
tf.flags.DEFINE_string("model", "BiLSTM", "Type of classifiers. You have six choices: [LSTM, BiLSTM, LSTM-pool, BiLSTM-pool, ATT-LSTM, ATT-BiLSTM] (default: LSTM)")
tf.flags.DEFINE_integer("hidden_layer_num", 1, "LSTM hidden layer num (default: 1)")
tf.flags.DEFINE_integer("hidden_neural_size", 100, "LSTM hidden neural size (default: 128)")
tf.flags.DEFINE_integer("attention_size", 200, "LSTM hidden neural size (default: 128)")

tf.flags.DEFINE_float("lr", 0.001, "learning rate (default=0.001)")
tf.flags.DEFINE_float("lr_decay", 0.9, "Learning rate decay rate (default: 0.98)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)") #살리는 확률
tf.flags.DEFINE_float("l2_reg_lambda", 1.0e-4, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("vocab_size", 30000, "Vocabulary size (defualt: 0)")
tf.flags.DEFINE_integer("num_classes", 0, "Number of classes (defualt: 0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps(literations) (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)") # early stop check point

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess():
    # Data Preparation
    # Load data
    print("Loading data...")
    x_text, y, lengths = dh.load_data(FLAGS.x_train_file, FLAGS.t_train_file)

    print("Build vocabulary...")
    # Build vocabulary
    word_id_dict, _ = dh.buildVocab(x_text, FLAGS.vocab_size)
    print(word_id_dict)
    FLAGS.vocab_size = len(word_id_dict) + 4
    print("vocabulary size: ", FLAGS.vocab_size)

    for word_id in word_id_dict.keys():
        word_id_dict[word_id] += 4  # 0: <pad>, 1: <unk>, 2: <s>
    word_id_dict['<pad>'] = 0
    word_id_dict['<unk>'] = 1
    word_id_dict['<s>'] = 2
    word_id_dict['</s>'] = 3

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_text = x_text[shuffle_indices]
    print("Split train/validation set...")
    val_sample_index = -1 * int(FLAGS.val_sample_percentage * float(len(y)))
    x_train, x_val = x_text[:val_sample_index], x_text[val_sample_index:]

    x_train = dh.text_to_index(x_train, word_id_dict, 0)
    x_val = dh.text_to_index(x_val, word_id_dict, 0)

    FLAGS.num_classes = y.shape[1]

    y = y[shuffle_indices]
    lengths = lengths[shuffle_indices]

    y_train, y_val = y[:val_sample_index], y[val_sample_index:]
    lengths, lengths_val = lengths[:val_sample_index], lengths[val_sample_index:]

    print("Vocabulary Size: {:d}".format(FLAGS.vocab_size))
    print("Train/Val split: {:d}/{:d}".format(len(y_train), len(y_val)))
    return x_train, y_train, lengths, word_id_dict, x_val, y_val, lengths_val

def train(x_train, y_train, lengths, word_id_dict, x_val, y_val, lengths_val):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            rnn = TextRNN(FLAGS.flag_values_dict())

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            decayed_lr = tf.train.exponential_decay(FLAGS.lr, global_step, 1000, FLAGS.lr_decay, staircase=True)
            optimizer = tf.train.AdamOptimizer(decayed_lr)
            grads_and_vars = optimizer.compute_gradients(rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            with smart_open.smart_open(os.path.join(out_dir, "vocab"), 'wb') as f:
                pickle.dump(word_id_dict, f)
            with smart_open.smart_open(os.path.join(out_dir, "config"), 'wb') as f:
                pickle.dump(FLAGS.flag_values_dict(), f)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if FLAGS.word2vec:
                print("Loading W2V data...")
                pre_emb = KeyedVectors.load_word2vec_format(FLAGS.word2vec, binary=True)
                pre_emb.init_sims(replace=True)
                num_keys = len(pre_emb.vocab)
                print("loaded word2vec len ", num_keys)

                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25, (FLAGS.vocab_size, FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("init initW cnn.W in FLAG")
                for w in word_id_dict.keys():
                    arr = []
                    s = re.sub('[^0-9a-zA-Z]+', '', w)
                    if w in pre_emb:
                        arr = pre_emb[w]
                    elif w.lower() in pre_emb:
                        arr = pre_emb[w.lower()]
                    elif s in pre_emb:
                        arr = pre_emb[s]
                    elif s.isdigit():
                        arr = pre_emb['1']
                    if len(arr) > 0:
                        idx = word_id_dict[w]
                        initW[idx] = np.asarray(arr).astype(np.float32)
                print("assigning initW to cnn. len=" + str(len(initW)))
                sess.run(rnn.W.assign(initW))

            def train_step(x_batch, y_batch, sequence_batch):
                """
                A single training step
                """
                feed_dict = {
                  rnn.input_x: dh.batch_tensor(x_batch),
                  rnn.input_y: y_batch,
                  rnn.sequence_length: sequence_batch,
                  rnn.batch_size: len(y_batch),
                  rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, lr, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, decayed_lr, train_summary_op, rnn.loss, rnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, lr {:g}, acc {:g}".format(time_str, step, loss, lr, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, sequence_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  rnn.input_x: dh.batch_tensor(x_batch),
                  rnn.input_y: y_batch,
                  rnn.sequence_length: sequence_batch,
                  rnn.batch_size: len(y_batch),
                  rnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, rnn.loss, rnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy

            # Generate batches
            batches = dh.batch_iter(
                list(zip(x_train, y_train, lengths)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            max = 0

            for batch in batches:
                x_batch, y_batch, length_batch = zip(*batch)
                train_step(x_batch, y_batch, length_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    accuracy = dev_step(x_val, y_val, lengths_val, writer=dev_summary_writer)
                    print("")
                    if accuracy > max:
                        max = accuracy
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, lengths, word_id_dict, x_val, y_val, lengths_val = preprocess()
    train(x_train, y_train, lengths, word_id_dict, x_val, y_val, lengths_val)

if __name__ == '__main__':
    tf.app.run()
