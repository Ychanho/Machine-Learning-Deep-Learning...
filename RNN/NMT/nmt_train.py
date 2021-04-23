
#!/usr/bin/env python
# coding: utf-8

import os
import math
import time
import json
import random
import datetime

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import smart_open
import pickle
from lecture_12 import data_helpers as dh
from lecture_12.nmt_rnn import NMTRNN

# Data loading parameters
tf.app.flags.DEFINE_string('source_train_data', 'data/korean-english-park.train.en', 'Path to source training data')
tf.app.flags.DEFINE_string('target_train_data', 'data/korean-english-park.train_stem.ko', 'Path to target training data')
tf.app.flags.DEFINE_string('source_dev_data', 'data/korean-english-park.dev.en', 'Path to source validation data')
tf.app.flags.DEFINE_string('target_dev_data', 'data/korean-english-park.dev_stem.ko', 'Path to target validation data')

# Network parameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of word embedding (default: 128)") # 현재는 랜덤하게 하는거야
tf.flags.DEFINE_string("model", "LSTM", "Type of classifiers. You have three choices: [LSTM, BiLSTM, LSTM-pool, BiLSTM-pool, ATT-LSTM, ATT-BiLSTM] (default: LSTM)")
tf.flags.DEFINE_string("infer_mode", "greedy", "Type of search argorithm. You have two choices: [greedy, beamsearch] (default: beamsearch)")
tf.flags.DEFINE_integer("hidden_layer_num", 1, "LSTM hidden layer num (default: 1)")
tf.flags.DEFINE_integer("hidden_neural_size", 1000, "LSTM hidden neural size (default: 128)")
tf.flags.DEFINE_integer("attention_size", 200, "LSTM hidden neural size (default: 128)")
tf.flags.DEFINE_integer("beam_size", 3, "beam search width (default: 3)")

tf.flags.DEFINE_float("lr", 0.001, "learning rate (default=0.001)")
tf.flags.DEFINE_float("lr_decay", 0.9, "Learning rate decay rate (default: 0.98)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability (default: 0.5)") #살리는 확률
tf.flags.DEFINE_float("l2_reg_lambda", 1.0e-5, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("source_vocab_size", 30000, "Vocabulary size (defualt: 0)")
tf.flags.DEFINE_integer("target_vocab_size", 30000, "Vocabulary size (defualt: 0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 300, "Evaluate model on dev set after this many steps(literations) (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 300, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)") # early stop check point

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess():
    # Load data
    print("Loading data...")
    x_text, t_text = dh.load_data(FLAGS.source_train_data, FLAGS.target_train_data)
    xv_text, tv_text = dh.load_data(FLAGS.source_dev_data, FLAGS.target_dev_data)

    print("Build vocabulary...")
    source_word_id_dict, _ = dh.buildVocab(x_text, FLAGS.source_vocab_size) # apple:1017 <--> 1017:apple
    FLAGS.source_vocab_size = len(source_word_id_dict) + 4
    target_word_id_dict, _ = dh.buildVocab(t_text, FLAGS.target_vocab_size)
    print(target_word_id_dict)
    FLAGS.target_vocab_size = len(target_word_id_dict) + 4
    print("Source language vocabulary size: ", FLAGS.source_vocab_size)
    print("Target language vocabulary size: ", FLAGS.target_vocab_size)
    print("Average length (source): ", sum([len(x.split(" ")) for x in x_text]) / len(x_text))
    print("Average length (target): ", sum([len(t.split(" ")) for t in t_text])/ len(t_text))
    # FLAGS.max_length = 2 + max([len(x.split(" ")) for x in x_text] + [len(t.split(" ")) for t in t_text])
    for word_id in source_word_id_dict.keys():
        source_word_id_dict[word_id] += 4  # 0: <pad>, 1: <s>, 2: <eos>, 3: <unk>
    source_word_id_dict['<pad>'] = 0
    source_word_id_dict['<s>'] = 1
    source_word_id_dict['<eos>'] = 2  # end of sentence
    source_word_id_dict['<unk>'] = 3  # Gooooooooood!!!!!! -> <unk>

    for word_id in target_word_id_dict.keys():
        target_word_id_dict[word_id] += 4
    target_word_id_dict['<pad>'] = 0
    target_word_id_dict['<s>'] = 1
    target_word_id_dict['<eos>'] = 2
    target_word_id_dict['<unk>'] = 3

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(x_text)))
    x_text = x_text[shuffle_indices]    # 문장 섞기
    t_text = t_text[shuffle_indices]

    x, x_lengths = dh.text_to_index(x_text, source_word_id_dict)
    t, t_lengths = dh.text_to_index_ko(t_text, target_word_id_dict)

    xv, xv_lengths = dh.text_to_index(xv_text, source_word_id_dict)
    tv, tv_lengths = dh.text_to_index_ko(tv_text, target_word_id_dict)

    print("Train/Val split: {:d}/{:d}".format(len(x), len(xv)))
    return x, t, x_lengths, t_lengths, xv, tv, xv_lengths, tv_lengths, source_word_id_dict, target_word_id_dict

def train(x, t, x_lengths, t_lengths, xv, tv, xv_lengths, tv_lengths, source_word_id_dict, target_word_id_dict):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            nmt = NMTRNN(FLAGS)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            decayed_lr = tf.train.exponential_decay(FLAGS.lr, global_step, 1000, FLAGS.lr_decay, staircase=True)
            optimizer = tf.train.AdamOptimizer(decayed_lr)
            grads_and_vars = optimizer.compute_gradients(nmt.loss)
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
            loss_summary = tf.summary.scalar("loss", nmt.loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            with smart_open.smart_open(os.path.join(out_dir, "source_vocab"), 'wb') as f:
                pickle.dump(source_word_id_dict, f)
            with smart_open.smart_open(os.path.join(out_dir, "target_vocab"), 'wb')as f:
                pickle.dump(target_word_id_dict, f)

            target_id_word_dict = {}
            for word in target_word_id_dict.keys():
                target_id_word_dict[target_word_id_dict[word]] = word

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch, x_length, y_length):

                feed_dict = {
                  nmt.input_x: dh.batch_tensor(x_batch), # 7, 123, 21, 0, 0, 0, 0, 0, 0, 0
                  nmt.y: dh.batch_tensor(y_batch),
                  nmt.source_length: x_length,
                  nmt.target_length: y_length,
                  nmt.batch_size: len(y_batch),
                  nmt.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, lr, summaries, loss, predictions, translation = sess.run(
                    [train_op, global_step, decayed_lr, train_summary_op, nmt.loss, nmt.predictions, nmt.translation],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, lr {:g}".format(time_str, step, loss, lr))
                print("source indices: ", x_batch[0])
                print("target indices: ", y_batch[0])
                print("prediction indices: ", predictions[0])
                print("prediction text: ", end=" ")
                for i in range(len(predictions[0])):
                    print(target_id_word_dict[predictions[0][i]], end=" ")
                print("")
                print("Inference text: ", end=" ")
                for i in range(len(translation[0])):
                    print(target_id_word_dict[translation[0][i]], end=" ")
                print("")

                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, x_length, y_length, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  nmt.input_x: x_batch,
                  nmt.y: y_batch,
                  nmt.source_length: x_length,
                  nmt.target_length: y_length,
                  nmt.batch_size: len(y_batch),
                  nmt.dropout_keep_prob: 1.0
                }
                step, summaries, loss = sess.run(
                    [global_step, dev_summary_op, nmt.loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
                if writer:
                    writer.add_summary(summaries, step)
                return loss

            # Generate batches
            batches = dh.batch_iter(list(zip(x, x_lengths, t, t_lengths)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            min = 1000000

            for batch in batches:
                x_batch, x_length, y_batch, y_length = zip(*batch)
                train_step(x_batch, y_batch, x_length, y_length)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    # loss = dev_step(xv, tv, xv_lengths, tv_lengths, writer=dev_summary_writer)
                    # print("")
                    # if loss < min:
                    #     min = loss
                    #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):    # 전처리
    x, t, x_lengths, t_lengths, xv, tv, xv_lengths, tv_lengths, source_word_id_dict, target_word_id_dict = preprocess()
    train(x, t, x_lengths, t_lengths, xv, tv, xv_lengths, tv_lengths, source_word_id_dict, target_word_id_dict)

if __name__ == '__main__':
    tf.app.run()
