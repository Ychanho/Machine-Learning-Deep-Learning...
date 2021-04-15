import tensorflow as tf


class TextRNN(object):

    def __init__(self, config):
        self.num_classes = config["num_classes"] # e.g., positive, negatie - 2
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_neural_size"]
        self.attention_size = config["attention_size"]
        self.embedding_dim = config["embedding_dim"] # word vector size
        self.num_layers = config["hidden_layer_num"] #
        self.l2_reg_lambda = config["l2_reg_lambda"]

        self.batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")  # [batch, s_length]
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")

        self.l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0), trainable=True, name="W")
            self.inputs = tf.nn.embedding_lookup(self.W, self.input_x)

        # LSTM
        if config["model"] == "LSTM":
            _, self.final_state = self.normal_lstm()
        elif config["model"] == "LSTM-pool":
            output, _ = self.normal_lstm()
            masks = tf.sequence_mask(lengths=self.sequence_length,
                                     maxlen=tf.reduce_max(self.sequence_length), dtype=tf.float32, name='masks')
            # 1.0 1.0 1.0 0.0 0.0
            output = output * tf.expand_dims(masks, -1)  # (batch, max_len, hidden_dim) (batch, max_len, 1)
            self.final_state = tf.div(tf.reduce_sum(output, 1), tf.expand_dims(tf.cast(self.sequence_length, tf.float32), 1))
        elif config["model"] == "BiLSTM":
            _, self.final_state = self.bi_lstm()
        elif config["model"] == "BiLSTM-pool":
            output, _ = self.bi_lstm()
            masks = tf.sequence_mask(lengths=self.sequence_length,
                                     maxlen=tf.reduce_max(self.sequence_length), dtype=tf.float32, name='masks')
            output_fw = output[0] * tf.expand_dims(masks, -1)
            output_bw = output[1] * tf.expand_dims(masks, -1)
            output_fw = tf.div(tf.reduce_sum(output_fw, 1), # batch, seq, hidden_size
                               tf.expand_dims(tf.cast(self.sequence_length, tf.float32), 1))
            output_bw = tf.div(tf.reduce_sum(output_bw, 1), # batch, seq, hidden_size
                               tf.expand_dims(tf.cast(self.sequence_length, tf.float32), 1))
            self.final_state = tf.concat([output_fw, output_bw], 1)

        self.final_state = tf.nn.dropout(self.final_state, keep_prob=self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            if config["model"] == "LSTM" or config["model"] == "LSTM-pool":
                W = tf.get_variable("W", shape=[self.hidden_size, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            else:
                W = tf.get_variable("W", shape=[self.hidden_size*2, self.num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.final_state, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def normal_lstm(self):
        # LSTM Cell
        cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True), output_keep_prob=self.dropout_keep_prob)
        # Stacked LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, tf.float32)

        # Dynamic LSTM
        with tf.variable_scope("LSTM"):
            output, final_state = tf.nn.dynamic_rnn(cell, inputs=self.inputs, initial_state=self._initial_state,
                                              sequence_length=self.sequence_length)
        print(output.shape) # batch, seq_length, hidden_size
        print(final_state[0].h.shape) # batch, hidden_size
        return output, final_state[0].h

    def bi_lstm(self):
        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True,
                                          reuse=tf.get_variable_scope().reuse)
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True,
                                          reuse=tf.get_variable_scope().reuse)

        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)

        self._initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        self._initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope("BiLSTM"):
            output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=self.inputs,
                                                       initial_state_fw=self._initial_state_fw,
                                                       initial_state_bw=self._initial_state_bw,
                                                       sequence_length=self.sequence_length)

            state_fw = state[0]
            state_bw = state[1]

            final_state = tf.concat([state_fw[self.num_layers - 1], state_bw[self.num_layers - 1]], 1)
            return output, final_state
