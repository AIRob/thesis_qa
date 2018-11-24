import tensorflow as tf
import numpy as np
from dyrnn import last_relevant,seq_length

class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """
    
    def BiRNN(self, x, dropout, scope,embedding_size,sequence_length, hidden_units):
        n_hidden=hidden_units
        n_layers=3
        # Prepare data shape to match `static_rnn` function requirements
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        # Get lstm cell output

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x,sequence_length=sequence_length,dtype=tf.float32)
            #outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x,sequence_length=sequence_length,dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs
    
    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        loss = tf.reduce_sum(tmp +tmp2)
        return  loss/batch_size
    
    def __init__(
        self, max_sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, max_sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, max_sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")
          
        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True,name="W")
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            #self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            #self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.len1 = seq_length(self.input_x1)
            self.len2 = seq_length(self.input_x2)
            self.out1=self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size,self.len1 , hidden_units)
            self.out2=self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size,self.len2, hidden_units)
            self.last_out1,self.last_out2 = last_relevant(self.out1,self.len1),last_relevant(self.out2,self.len2)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.last_out1,self.last_out2)),1,keep_dims=True))
            self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.last_out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.last_out2),1,keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")

        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
