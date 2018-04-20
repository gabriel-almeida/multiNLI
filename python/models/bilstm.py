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

        def multiplicative_attention(query, hidden, sizes, hidden_dim, max_size, namespace, reuse):
            with tf.variable_scope('attention_' + namespace, reuse=reuse):
               att_w = tf.get_variable("weight", shape=[hidden_dim, hidden_dim], 
                                       dtype=tf.float32)
               A = tf.matmul(query, att_w)
               B = tf.matmul(hidden, tf.expand_dims(A, 2))
               B = tf.squeeze(B)

               mask = tf.sequence_mask(sizes, max_size, dtype=tf.float32)
               exp_B = tf.exp(B)*mask
               denominator = tf.reduce_sum(exp_B, axis=1)
               softmax = tf.div(exp_B, tf.expand_dims(denominator, 1))
               weight_sum = tf.reduce_sum(tf.multiply(hidden, 
                                                      tf.expand_dims(softmax, 2)), axis=1)
               return weight_sum, softmax
  

        def network(premise, hypothesis, reuse=False):
            # Get lengths of unpadded sentences
            prem_seq_lengths, prem_mask = blocks.length(premise)
            hyp_seq_lengths, hyp_mask = blocks.length(hypothesis)

            ### BiLSTM layer ###
            premise_in = emb_drop(premise)
            hypothesis_in = emb_drop(hypothesis)

            premise_outs, c1 = blocks.biLSTM(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='premise',
                                             reuse=reuse)
            hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis',
                                                reuse=reuse)

            premise_bi = tf.concat(premise_outs, axis=2)
            hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

            #premise_final = blocks.last_output(premise_bi, prem_seq_lengths)
            #hypothesis_final =  blocks.last_output(hypothesis_bi, hyp_seq_lengths)

            ### Mean pooling
            premise_sum = tf.reduce_sum(premise_bi, 1)
            premise_ave_old = tf.div(premise_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1))

            hypothesis_sum = tf.reduce_sum(hypothesis_bi, 1)
            hypothesis_ave_old = tf.div(hypothesis_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1))
            
            # premise_ave, _ = multiplicative_attention(hypothesis_ave_old, premise_bi, 
            #                                          prem_seq_lengths, self.dim*2, 
            #                                          self.sequence_length, "premise", reuse=reuse)

            # hypothesis_ave, _ = multiplicative_attention(premise_ave_old, hypothesis_bi, 
            #                                             hyp_seq_lengths, self.dim*2, 
            #                                             self.sequence_length, "hypothesis", reuse=reuse)

            premise_ave = premise_ave_old
            hypothesis_ave = hypothesis_ave_old

            ### Mou et al. concat layer ###
            diff = tf.subtract(premise_ave, hypothesis_ave)
            mul = tf.multiply(premise_ave, hypothesis_ave)
            h = tf.concat([premise_ave, hypothesis_ave, diff, mul], 1)

            # MLP layer
            h_mlp = tf.nn.relu(tf.matmul(h, self.W_mlp) + self.b_mlp)
            # Dropout applied to classifier
            h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

            # Get prediction
            return tf.matmul(h_drop, self.W_cl) + self.b_cl

        self.logits = network(self.premise_x, self.hypothesis_x)
        self.reversed_logits = network(self.hypothesis_x, self.premise_x, reuse=True)

        self.original_probs = tf.nn.softmax(self.logits)
        self.reverse_probs = tf.nn.softmax(self.reversed_logits)

        # semi supervised draft

        # supervised_idx = tf.not_equal(self.y, tf.constant(-1))
        #
        # supervised_logits = tf.boolean_mask(self.logits, supervised_idx)
        # supervised_y = tf.boolean_mask(self.y, supervised_idx)
        # total_supervision = tf.reduce_sum(tf.cast(supervised_idx, dtype=tf.float32))
        #
        # self.total_cost = tf.cond(tf.greater_equal(total_supervision, tf.constant(1.0)),
        #                           lambda: tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #                               labels=supervised_y, logits=supervised_logits)) / total_supervision,
        #                           lambda: tf.constant(0.0))

        # Define the cost function
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y, logits=self.logits))

        self.inference_value = tf.reduce_mean(logic_regularizer.fuzzy_inference(self.original_probs,
                                                                                self.reverse_probs))
        self.neutral_value = tf.reduce_mean(logic_regularizer.fuzzy_neutral(self.original_probs,
                                                                            self.reverse_probs))
        self.contradiction_value = tf.reduce_mean(logic_regularizer.semantic_contradiction(self.original_probs,
                                                                                        self.reverse_probs))

        #self.only_one_original_value = tf.reduce_mean(logic_regularizer.semantic_only_one(self.original_probs))
        #self.only_one_reversed_value = tf.reduce_mean(logic_regularizer.semantic_only_one(self.reverse_probs))

        #self.regularized_loss = self.total_cost + self.pi * (self.inference_value + 2.0*self.contradiction_value + self.neutral_value)/4.0
        self.regularized_loss = self.inference_value + self.contradiction_value + self.neutral_value
