import tensorflow as tf
from util import logic_regularizer
from util import blocks

class MyModel(object):
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings, emb_train):
        ## Define hyperparameters
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length 

        ## Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])
        self.pi = tf.placeholder(tf.float32, None)

        ## Define parameters
        # self.E = tf.Variable(embeddings, trainable=emb_train)

        with tf.device('/cpu:0'):
            self.E = tf.Variable(tf.random_uniform(embeddings.shape, -1.0,1.0),
                        trainable=emb_train, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, embeddings.shape)
            self.embedding_init = self.E.assign(self.embedding_placeholder)
        
        self.W_mlp = tf.Variable(tf.random_normal([self.dim * 8, self.dim], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))
        
        ## Function for embedding lookup and dropout at embedding layer
        def emb_drop(x):
            emb = tf.nn.embedding_lookup(self.E, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return emb_drop

        # Get lengths of unpadded sentences
        prem_seq_lengths, prem_mask = blocks.length(self.premise_x)
        hyp_seq_lengths, hyp_mask = blocks.length(self.hypothesis_x)


        ### BiLSTM layer ###
        premise_in = emb_drop(self.premise_x)
        hypothesis_in = emb_drop(self.hypothesis_x)

        premise_outs, c1 = blocks.biLSTM(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='premise')
        hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')

        premise_bi = tf.concat(premise_outs, axis=2)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

        #premise_final = blocks.last_output(premise_bi, prem_seq_lengths)
        #hypothesis_final =  blocks.last_output(hypothesis_bi, hyp_seq_lengths)

        ### Mean pooling
        premise_sum = tf.reduce_sum(premise_bi, 1)
        premise_ave = tf.div(premise_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1))

        hypothesis_sum = tf.reduce_sum(hypothesis_bi, 1)
        hypothesis_ave = tf.div(hypothesis_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1))

        ### Mou et al. concat layer ###
        diff = tf.subtract(premise_ave, hypothesis_ave)
        mul = tf.multiply(premise_ave, hypothesis_ave)
        h = tf.concat([premise_ave, hypothesis_ave, diff, mul], 1)

        # MLP layer
        h_mlp = tf.nn.relu(tf.matmul(h, self.W_mlp) + self.b_mlp)
        # Dropout applied to classifier
        h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

        ##################

        diff2 = tf.subtract(hypothesis_ave, premise_ave)
        h = tf.concat([hypothesis_ave, premise_ave, diff2, mul], 1)
        h_mlp = tf.nn.relu(tf.matmul(h, self.W_mlp) + self.b_mlp)
        h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

        self.reverse_probs = tf.nn.softmax(tf.matmul(h_drop, self.W_cl) + self.b_cl)
        self.original_probs = tf.nn.softmax(self.logits)

        ##################

        # Define the cost function
        supervised_idx = tf.not_equal(self.y, -1)

        supervised_logits = self.logits[:, supervised_idx]
        supervised_y = self.y[:, supervised_idx]
        total_supervision = tf.reduce_sum(tf.cast(supervised_idx, dtype=tf.float32))

        self.total_cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=supervised_y, logits=supervised_logits))
        self.total_cost = self.total_cost / total_supervision

        self.inference_value = tf.reduce_mean(
            -tf.log(logic_regularizer.inference_rule(self.original_probs, self.reverse_probs) + 0.0001))
        self.contradiction_value = tf.reduce_mean(
            -tf.log(logic_regularizer.contradiction_rule(self.original_probs, self.reverse_probs) + 0.0001))

        self.regularized_loss = self.total_cost + self.pi * (self.inference_value + self.contradiction_value)
