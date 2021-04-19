import tensorflow as tf
import numpy as np


class NMTRNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, config):
        self.source_vocab_size = config.source_vocab_size
        self.target_vocab_size = config.target_vocab_size
        self.hidden_size = config.hidden_neural_size
        self.attention_size = config.attention_size
        self.embedding_dim = config.embedding_dim # word vector size
        self.num_layers = config.hidden_layer_num #
        self.l2_reg_lambda = config.l2_reg_lambda
        self.infer_mode = config.infer_mode
        self.beam_size = config.beam_size

        self.batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x") # (batch, seq) I love you -> 7, 123, 21
        self.y = tf.placeholder(tf.int32, [None, None], name="input_y") # (batch, seq) 나는 너를 사랑해 -> 9, 201, 14
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.source_length = tf.placeholder(tf.int32, [None], name="source_length")#(batch)
        self.target_length = tf.placeholder(tf.int32, [None], name="target_length")#(batch)

        decoder_start_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * 1
        decoder_end_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * 2
        self.input_y = tf.concat([decoder_start_token, self.y], axis=1)  # <s> ...
        self.output_y = tf.concat([self.y, decoder_end_token], axis=1)  #... <eos>
        print("output_y: ", self.output_y)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.EW = tf.Variable(tf.random_uniform([self.source_vocab_size, self.embedding_dim], -1.0, 1.0), trainable=True, name="EW")
            self.inputs = tf.nn.embedding_lookup(self.EW, self.input_x)
            print("inputs:", self.inputs.get_shape())

            self.DW = tf.Variable(tf.random_uniform([self.target_vocab_size, self.embedding_dim], -1.0, 1.0),trainable=True, name="DW")
            self.decoder_inputs = tf.nn.embedding_lookup(self.DW, self.input_y)
            print("decoder_inputs:", self.decoder_inputs.get_shape())

        self.output_layer = tf.layers.Dense(self.target_vocab_size, use_bias=False, name="output_projection")

        self.decoder_inputs = tf.nn.dropout(self.decoder_inputs, keep_prob=self.dropout_keep_prob)

        # LSTM encoder
        if config.model == "LSTM":
            outputs, self.encoder_state = self.encoder()
            # self.encoder_state = tf.div(tf.reduce_sum(outputs, 1), tf.expand_dims(tf.cast(self.source_length, tf.float32), 1))

        print("encoder state:", self.encoder_state)

        # LSTM decoder
        if config.model == "LSTM":
            self.logits = self.decoder()
        print("logits shape: ", self.logits.get_shape()) # (batch, seq, hidden_dim)
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):

            costs = []
            for var in tf.trainable_variables():
                costs.append(tf.nn.l2_loss(var))
            l2_loss = tf.add_n(costs)

            self.decoder_logits_train = tf.identity(self.logits)
            masks = tf.sequence_mask(lengths=self.target_length,
                                     maxlen=tf.reduce_max(self.target_length), dtype=tf.float32, name='masks')

            losses = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                              targets=self.output_y,
                                              weights=masks,
                                              average_across_timesteps=True,
                                              average_across_batch=True,)

            # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.output_y)
            # target_weights = tf.sequence_mask(self.target_length, self.max_length, dtype=tf.float32)
            self.predictions = tf.argmax(self.logits, 2, name="predictions")
            self.loss = losses + self.l2_reg_lambda * l2_loss

        with tf.name_scope("inference"):
            self.translation = tf.reshape(self.inference(), [self.batch_size, -1], name="translation")

    def encoder(self):
        # LSTM Cell
        # cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True), output_keep_prob=self.dropout_keep_prob)
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True)
        # Stacked LSTMs
        # cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(self.batch_size, tf.float32)
        # Dynamic LSTM
        with tf.variable_scope("LSTM"):
            output, final_state = tf.nn.dynamic_rnn(cell, inputs=self.inputs, sequence_length=self.source_length, initial_state=self._initial_state)

        return output, final_state

    def decoder(self):
        # LSTM Cell
        self.decoder_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True),
                                             output_keep_prob=self.dropout_keep_prob)
        print("target length: ", self.target_length.get_shape())
        # print("output layer: ", self.output_layer)
        helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_inputs, self.target_length)
        decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, self.encoder_state, output_layer=self.output_layer)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=tf.reduce_max(self.target_length))

        return outputs.rnn_output

    def inference(self):
        def embed_and_input_proj(inputs):
            return tf.nn.embedding_lookup(self.DW, inputs)

        if self.infer_mode == "beamsearch":
            self.encoder_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, multiplier=self.beam_size)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.decoder_cell,
                                                                      embedding=embed_and_input_proj,
                                                                      start_tokens=tf.fill([self.batch_size], 1), #<s>
                                                                      end_token=2,
                                                                      initial_state=self.encoder_state,
                                                                      beam_width=self.beam_size,
                                                                      output_layer=self.output_layer)
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=40)

            return outputs.predicted_ids

        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embed_and_input_proj, start_tokens=tf.fill([self.batch_size], 1), end_token=2)

            decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, self.encoder_state,
                                                      output_layer=self.output_layer)
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=40)

            return outputs.sample_id

