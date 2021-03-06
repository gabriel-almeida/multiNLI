"""
Training script to train a model on MultiNLI and, optionally, on SNLI data as well.
The "alpha" hyperparamaters set in paramaters.py determines if SNLI data is used in training. 
If alpha = 0, no SNLI data is used in training. If alpha > 0, then down-sampled SNLI data is used in training. 
"""

import tensorflow as tf
import os
import importlib
import time
import random
import math
import multiprocessing
from util import logic_regularizer
from util import logger
import util.parameters as params
from util.data_processing import *
from util.evaluate import *

FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS["model_type"]

module = importlib.import_module(".".join(['models', model])) 
MyModel = getattr(module, 'MyModel')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistently use the same hyperparameter settings. 
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


######################### LOAD DATA #############################

logger.Log("Loading data")
training_snli = load_nli_data(FIXED_PARAMETERS["training_snli"], snli=True)
dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"], snli=True)

logger.Log("Loaded sentences SNLI - Train: %s | Dev: %s | Test: %s" % (len(training_snli), len(dev_snli), len(test_snli)))

training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"])
dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])
test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"])
test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"])

logger.Log("Loaded sentences MultiNLI - Train: %s | Matched Dev: %s | Mismatched Dev: %s | Matched Test: %s | Mismatched Test: %s" % (len(training_mnli),\
           len(dev_matched), len(dev_mismatched), len(test_matched), len(test_mismatched)))

if 'temp.jsonl' in FIXED_PARAMETERS["test_matched"]:
    # Removing temporary empty file that was created in parameters.py
    os.remove(FIXED_PARAMETERS["test_matched"])
    logger.Log("Created and removed empty file called temp.jsonl since test set is not available.")

dictpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".p"

if not os.path.isfile(dictpath): 
    logger.Log("Building dictionary")
    if FIXED_PARAMETERS["alpha"] == 0:
        word_indices = build_dictionary([training_mnli])
    else:
        word_indices = build_dictionary([training_mnli, training_snli])
    
    logger.Log("Padding and indexifying sentences")
    sentences_to_padded_index_sequences(word_indices, [training_mnli, training_snli, 
                                        dev_matched, dev_mismatched, dev_snli, test_snli, 
                                        test_matched, test_mismatched])
    pickle.dump(word_indices, open(dictpath, "wb"))

else:
    logger.Log("Loading dictionary from %s" % (dictpath))
    word_indices = pickle.load(open(dictpath, "rb"))
    logger.Log("Padding and indexifying sentences")
    sentences_to_padded_index_sequences(word_indices, [training_mnli, training_snli, 
                                        dev_matched, dev_mismatched, dev_snli, 
                                        test_snli, test_matched, test_mismatched])

logger.Log("Loading embeddings")
loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)


class modelClassifier:
    def __init__(self, seq_length, loaded_embeddings):
        ## Define hyperparameters
        self.learning_rate =  FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_ratio = FIXED_PARAMETERS["display_step_ratio"]
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        n_samples = 0
        self.eval_batch_size = FIXED_PARAMETERS["eval_batch_size"]
        self.max_patience = FIXED_PARAMETERS["patience"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"] 
        self.alpha = FIXED_PARAMETERS["alpha"]
        self.word_embeddings = loaded_embeddings
        self.unsupervised_ratio = FIXED_PARAMETERS["unsupervised_ratio"]
        self.pi = FIXED_PARAMETERS["pi"]

        logger.Log("Building model from %s.py" %(model))
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim, 
                                hidden_dim=self.dim, embeddings=loaded_embeddings, 
                                emb_train=self.emb_train)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.model.regularized_loss)

        # Boolean stating that training has not been completed, 
        self.completed = False 

        # tf things: initialize variables and create placeholder for session
        logger.Log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    def get_minibatch(self, dataset, start_index, end_index):
        end_index = min(end_index, len(dataset))
        indices = range(start_index, end_index)
        premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
        genres = [dataset[i]['genre'] for i in indices]
        labels = [dataset[i]['label'] for i in indices]
        return premise_vectors, hypothesis_vectors, labels, genres

    def train(self, train_mnli, train_snli, dev_mat, dev_mismat, dev_snli):
        config = tf.ConfigProto()
        n_threads = int(multiprocessing.cpu_count() * 2)
        config.intra_op_parallelism_threads = n_threads
        config.inter_op_parallelism_threads = n_threads

        self.sess = tf.Session(config=config)
        self.sess.run(self.init)

        self.max_epochs = FIXED_PARAMETERS["epochs"]
        self.patience = self.max_patience
        self.step = 0
        self.epoch = 0
        self.best_dev_mat = 0.
        self.best_mtrain_acc = 0.
        self.display_step = None

        self.sess.run(self.model.embedding_init, feed_dict={self.model.embedding_placeholder: self.word_embeddings})

        # Restore most recent checkpoint if it exists. 
        # Also restore values for best dev-set accuracy and best training-set accuracy
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                self.best_dev_mat, dev_cost_mat, _ = evaluate_classifier(self.classify, dev_mat, self.eval_batch_size)
                best_dev_mismat, dev_cost_mismat, _ = evaluate_classifier(self.classify, dev_mismat, self.eval_batch_size)
                best_dev_snli, dev_cost_snli, _ = evaluate_classifier(self.classify, dev_snli, self.eval_batch_size)
                self.best_mtrain_acc, mtrain_cost, _ = evaluate_classifier(self.classify, train_mnli[0:5000], self.eval_batch_size)
                if self.alpha != 0.:
                    self.best_strain_acc, strain_cost, _ = evaluate_classifier(self.classify, train_snli[0:5000], self.eval_batch_size)
                    logger.Log("Restored best matched-dev acc: %f\n Restored best mismatched-dev acc: %f\n \
                            Restored best SNLI-dev acc: %f\n Restored best MulitNLI train acc: %f\n \
                            Restored best SNLI train acc: %f" %(self.best_dev_mat, best_dev_mismat, best_dev_snli, 
                            self.best_mtrain_acc, self.best_strain_acc))
                else:
                    logger.Log("Restored best matched-dev acc: %f\n Restored best mismatched-dev acc: %f\n \
                         Restored best SNLI-dev acc: %f\n Restored best MulitNLI train acc: %f" 
                         % (self.best_dev_mat, best_dev_mismat, best_dev_snli, self.best_mtrain_acc))

            self.saver.restore(self.sess, ckpt_file)
            logger.Log("Model restored from file: %s" % ckpt_file)

        # Combine MultiNLI and SNLI data. Alpha has a default value of 0, if we want to use SNLI data, it must be passed as an argument.
        beta = int(self.alpha * len(train_snli))

        ### Training cycle
        logger.Log("Training...")
        logger.Log("Model will use %s percent of SNLI data during training" %(self.alpha * 100))

        contradiction_values = []
        neutral_values = []
        inference_values = []
        batch_times = []
        loss_values = []
        regularized_loss = []
        only_one_original_loss = []
        only_one_reversed_loss = []

        total_labels = 0
        for val in train_mnli:
            if random.random() < self.unsupervised_ratio:
                val['label'] = -1
            else:
                total_labels = total_labels + 1
        logger.Log("Total labels: " + str(total_labels))

        for self.epoch in range(self.max_epochs):
            training_data = train_mnli + random.sample(train_snli, beta)
            random.shuffle(training_data)
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)
            if len(training_data) % self.batch_size != 0:
                total_batch += 1

            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres = self.get_minibatch(
                    training_data, self.batch_size * i, self.batch_size * (i + 1))

                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function for logging

                #current_pi = min(1.0*self.step/(2.0*total_batch), 1.0)*self.pi
                current_pi = self.pi
                feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                             self.model.hypothesis_x: minibatch_hypothesis_vectors,
                             self.model.y: minibatch_labels,
                             self.model.keep_rate_ph: self.keep_rate, self.model.pi: current_pi}

                begin_batch_time = time.time()
                _, c, regularized_loss_val, inference_val, contradiction_val, neutral_val = \
                    self.sess.run([self.optimizer, self.model.total_cost, self.model.regularized_loss,
                                   self.model.inference_value, self.model.contradiction_value, self.model.neutral_value], feed_dict)

                batch_time = time.time() - begin_batch_time
                batch_times += [batch_time]
                contradiction_values += [contradiction_val]
                neutral_values += [neutral_val]
                inference_values += [inference_val]
                loss_values += [c]
                regularized_loss += [regularized_loss_val]

                if self.display_step is None or (self.step % total_batch) % self.display_step == self.display_step - 1:
                    if self.step == 0:  # do not evaluate on first batch, because it tends to be much slower than average
                        self.step += 1
                        avg_cost += c / (total_batch * self.batch_size)
                        continue 
                    begin_eval_time = time.time()
                    dev_acc_mat, dev_cost_mat, dev_confusion, \
                    (valid_inference, total_inference), (valid_contradiction, total_contradiction), (valid_neutral, total_neutral), \
                    coherent_confusion, non_coherent_confusion, inversion_matrix = \
                        evaluate_classifier(self.classify, dev_mat, self.eval_batch_size, include_reverse=True)

                    dev_acc_mismat, dev_cost_mismat, dev_acc_snli, dev_cost_snli, mtrain_acc, mtrain_cost = (0., 0., 0., 0., 0., 0.)
                    eval_time = time.time() - begin_eval_time

                    if self.display_step is None:
                        optimal_display_step = eval_time/batch_time/self.display_step_ratio
                        n_evals_epoch = max(total_batch // optimal_display_step, 1)
                        # avoid evals next to the end of the epoch
                        self.display_step = int(math.floor(1.0*total_batch/n_evals_epoch))
                        logger.Log("Evaluating on every %s steps (%s times per epoch)" % (self.display_step, n_evals_epoch))

                    if self.alpha != 0.:
                        strain_acc, strain_cost, _ = evaluate_classifier(self.classify, train_snli[0:5000], self.eval_batch_size)
                        logger.Log("Step: %i\t Dev-matched acc: %f\t Dev-mismatched acc: %f\t \
                            Dev-SNLI acc: %f\t MultiNLI train acc: %f\t SNLI train acc: %f"
                                   % (self.step, dev_acc_mat, dev_acc_mismat, dev_acc_snli, mtrain_acc, strain_acc))
                        logger.Log("Step: %i\t Dev-matched cost: %f\t Dev-mismatched cost: %f\t \
                            Dev-SNLI cost: %f\t MultiNLI train cost: %f\t SNLI train cost: %f"
                                   % (self.step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, mtrain_cost, strain_cost))
                    else:
                        logger.Log("Step: %i\t Dev-matched acc: %f\t Dev-mismatched acc: %f\t \
                            Dev-SNLI acc: %f\t MultiNLI train acc: %f" %(self.step, dev_acc_mat,
                                                                         dev_acc_mismat, dev_acc_snli, mtrain_acc))
                        logger.Log("Step: %i\t Dev-matched cost: %f\t Dev-mismatched cost: %f\t \
                            Dev-SNLI cost: %f\t MultiNLI train cost: %f" %(self.step, dev_cost_mat,
                                                                           dev_cost_mismat, dev_cost_snli, mtrain_cost))

                    def statistic_log(name, values):
                        logger.Log("[epoch %s step %s] %s: Mean=%s Std=%s Min=%s Max=%s" % (self.epoch, self.step, name, np.mean(values), np.std(values), np.min(values), np.max(values)))

                    logger.Log("[epoch %s step %s] Confusion matrix on dev (target, predicted): %s" % (self.epoch, self.step, dev_confusion))
                    logger.Log("[epoch %s step %s] Coherent matrix on dev (target, predicted): %s" % (self.epoch, self.step, coherent_confusion))
                    logger.Log("[epoch %s step %s] Non Coherent  matrix on dev (target, predicted): %s" % (self.epoch, self.step, non_coherent_confusion))
                    logger.Log("[epoch %s step %s] Inversion matrix on dev (original, inverted): %s" % (self.epoch, self.step, inversion_matrix))

                    logger.Log("[epoch %s step %s] Dev inference rule: %s consistent / %s total = %s%%" % (
                    self.epoch, self.step, valid_inference, total_inference, 100.0*valid_inference/total_inference if total_inference != 0 else 0))
                    logger.Log("[epoch %s step %s] Dev contradiction rule: %s consistent / %s total = %s%%" % (
                        self.epoch, self.step, valid_contradiction, total_contradiction,
                        100.0*valid_contradiction/total_contradiction if total_contradiction != 0 else 0))

                    logger.Log("[epoch %s step %s] Dev neutral rule: %s consistent / %s total = %s%%" % (
                        self.epoch, self.step, valid_neutral, total_neutral,
                        100.0*valid_neutral/total_neutral if total_neutral != 0 else 0))

                    statistic_log("Contradiction value", contradiction_values)
                    statistic_log("Inference value", inference_values)
                    statistic_log("Neutral value", neutral_values)
                    statistic_log("Train loss", loss_values)
                    statistic_log("Regularized loss", regularized_loss)
                    statistic_log("Batch time", batch_times)
                    logger.Log("Evaluation time: %s" % (eval_time))
                    
                    batch_times = []
                    loss_values = []
                    contradiction_values = []
                    inference_values = []
                    regularized_loss = []

                    improvement_ratio = 100.0 * (1.0 - self.best_dev_mat / dev_acc_mat)
                    if improvement_ratio > 0.1:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_mat = dev_acc_mat
                        self.best_mtrain_acc = mtrain_acc
                        if self.alpha != 0.:
                            self.best_strain_acc = strain_acc
                        logger.Log("Checkpointing with new best matched-dev accuracy: %f (+%f %%)" %(self.best_dev_mat, improvement_ratio))
                        self.patience = self.max_patience
                    else:
                        self.patience -= 1
                        logger.Log("Reducing patience: %s" % self.patience)
                        if self.patience < 0:
                            break

                self.step += 1

                # Compute average loss
                avg_cost += c / (total_batch * self.batch_size)
                                
            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f" %(self.epoch+1, avg_cost))
            if self.patience < 0:
                break

        logger.Log("Best matched-dev accuracy: %s" %(self.best_dev_mat))
        logger.Log("MultiNLI Train accuracy: %s" %(self.best_mtrain_acc))
        self.completed = True

    def classify(self, examples, include_reverse=False):
        # This classifies a list of examples
        total_batch = int(len(examples) / self.eval_batch_size)
        if len(examples) % self.eval_batch_size != 0:
            total_batch = total_batch + 1
        logits = np.empty(3)
        genres = []
        mean_cost = 0
        reversed_probs = []
        for i in range(total_batch):
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres = \
                self.get_minibatch(examples, self.eval_batch_size * i, self.eval_batch_size * (i + 1))
            feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels, 
                                self.model.keep_rate_ph: 1.0}
            genres += minibatch_genres
            logit, cost, reversed_prob = self.sess.run([self.model.original_probs, self.model.total_cost,
                                                        self.model.reverse_probs], feed_dict)
            reversed_probs += [reversed_prob]
            mean_cost += 1.0/(i+1)*(cost - mean_cost)
            logits = np.vstack([logits, logit])

        reversed_probs = np.vstack(reversed_probs)

        if not include_reverse:
            return genres, np.argmax(logits[1:], axis=1), mean_cost
        else:
            return genres, np.argmax(logits[1:], axis=1), mean_cost, np.argmax(reversed_probs, axis=1), logits[1:], reversed_probs

    def restore(self, best=True):
        if best:
            path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
        else:
            path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        self.sess = tf.Session()
        self.saver.restore(self.sess, path)
        logger.Log("Model restored from file: %s" % path)


classifier = modelClassifier(FIXED_PARAMETERS["seq_length"], loaded_embeddings)

"""
Either train the model and then run it on the test-sets or 
load the best checkpoint and get accuracy on the test set. Default setting is to train the model.
"""

test = params.train_or_test()

if not test:
    classifier.train(training_mnli, training_snli, dev_matched, dev_mismatched, dev_snli)
else:
    logger.Log("Restoring model...")
    classifier.restore()

    def print_worst_logic(title, objs, dataset):
        logger.Log(title)
        idxs, probs, inv_probs, loss_vals = objs
        for i, (idx, prob, inv_prob, loss_val) in enumerate(zip(idxs, probs, inv_probs, loss_vals)):
            curr_data = dataset[idx]
            logger.Log("Case %s)" % (i+1))
            logger.Log("P(Inference)=%s | P(Neutral)=%s | P(Contradiction)=%s" % (prob[INFERENCE_CLASS], prob[NEUTRAL_CLASS], prob[CONTRADICTION_CLASS]))
            logger.Log("P(Inv_Inference)=%s | P(Inv_Neutral)=%s | P(Inv_Contradiction)=%s" % (inv_prob[INFERENCE_CLASS], inv_prob[NEUTRAL_CLASS], inv_prob[CONTRADICTION_CLASS]))
            logger.Log("Loss=%s" % loss_val)
            logger.Log("Sent1: %s" % (curr_data["sentence1"].encode('utf-8')))
            logger.Log("Sent2: %s" % (curr_data["sentence2"].encode('utf-8')))
        logger.Log("")

    def print_worst_case(title, objs, dataset):
        logger.Log(title)
        idxs, probs = objs
        for i, (idx, prob) in enumerate(zip(idxs, probs)):
            curr_data = dataset[idx]
            logger.Log("Case %s) P(Inference)=%s | P(Neutral)=%s | P(Contradiction)=%s" % (i+1, prob[INFERENCE_CLASS], prob[NEUTRAL_CLASS], prob[CONTRADICTION_CLASS]))
            logger.Log("Sent1: %s" % (curr_data["sentence1"].encode('utf-8')))
            logger.Log("Sent2: %s" % (curr_data["sentence2"].encode('utf-8')))
        logger.Log("")

    def evaluate(setname, dataset):
        acc_mat, cost_mat, confusion, \
        (valid_inference, total_inference), (valid_contradiction, total_contradiction), (valid_neutral, total_neutral), \
        coherent_confusion, non_coherent_confusion, inversion_matrix, \
        (worst_contradiction, worst_neutral, worst_inference), \
        (worst_logical_contradiction, worst_logical_neutral, worst_logical_inference) = \
            evaluate_classifier(classifier.classify, dataset, classifier.eval_batch_size, include_reverse=True, qualitative=True)
        logger.Log("Evaluating on %s:" % setname)
        logger.Log("-------------------")
        logger.Log("Accuracy: %s" % (acc_mat*100.0))
        logger.Log("Mean Loss: %s" % cost_mat)
        logger.Log("Total Confusion (target, predicted): %s" % confusion)
        logger.Log("Coherent Confusion (target, predicted): %s" % coherent_confusion)
        logger.Log("Non coherent confusion (target, predicted): %s" % non_coherent_confusion)
        logger.Log("Inversion matrix (original, inverted): %s" % inversion_matrix)
        logger.Log("Inference rule: %s consistent / %s total = %s%%" % (valid_inference, total_inference, 100.0*valid_inference/total_inference if total_inference != 0 else 0))
        logger.Log("Contradiction rule: %s consistent / %s total = %s%%" % (valid_contradiction, total_contradiction, 100.0*valid_contradiction/total_contradiction if total_contradiction != 0 else 0))
        logger.Log("Neutral rule: %s consistent / %s total = %s%%" % (valid_neutral, total_neutral, 100.0*valid_neutral/total_neutral if total_neutral != 0 else 0))

        print_worst_logic("Worsts inference logic", worst_logical_inference, dataset)
        print_worst_logic("Worsts neutral logic", worst_logical_neutral, dataset)
        print_worst_logic("Worsts contradiction logic", worst_logical_contradiction, dataset)


        print_worst_case("Worsts inference case", worst_inference, dataset)
        print_worst_case("Worsts neutral case", worst_neutral, dataset)
        print_worst_case("Worsts contradiction case", worst_contradiction, dataset)
        
        logger.Log("\n")
    evaluate("Train", training_mnli[:10000])
    evaluate("Dev Matched", dev_matched)
    evaluate("Dev Mismatched", dev_mismatched)
    evaluate("Dev SNLI", dev_snli)

    predictions_kaggle(classifier.classify, test_matched, modname + "_matched_hard")
    predictions_kaggle(classifier.classify, test_mismatched, modname + "_mismatched_hard")
