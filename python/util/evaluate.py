import csv
from util import logic_regularizer
import collections
import numpy as np
import tensorflow as tf

CONTRADICTION_CLASS = logic_regularizer.CONTRADICTION_CLASS
INFERENCE_CLASS = logic_regularizer.INFERENCE_CLASS
NEUTRAL_CLASS = logic_regularizer.NEUTRAL_CLASS

def evaluate_classifier(classifier, eval_set, batch_size, include_reverse=False, qualitative=False):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    correct = 0
    if not include_reverse:
        genres, hypotheses, cost = classifier(eval_set)
    else:
        genres, hypotheses, cost, reversed, probs, inverse_probs = classifier(eval_set, include_reverse=True)

    confusion_matrix = collections.Counter()
    confusion_matrix_coherent = collections.Counter()
    confusion_matrix_not_coherent = collections.Counter()
    inversion_matrix = collections.Counter()

    neutral_idx = []
    contradiction_idx = []
    inference_idx = [] 

    for i, predicted in enumerate(hypotheses):
        target = eval_set[i]['label']
        confusion_matrix.update([ (target, predicted) ])
        if include_reverse:
            if target == CONTRADICTION_CLASS: 
                contradiction_idx += [i]

            if target == NEUTRAL_CLASS: 
                neutral_idx += [i]
            
            if target == INFERENCE_CLASS: 
                inference_idx += [i]
            
            inversion_matrix.update([ (predicted, reversed[i]) ])
            if predicted == CONTRADICTION_CLASS and reversed[i] != CONTRADICTION_CLASS \
               or predicted != CONTRADICTION_CLASS and reversed[i] == CONTRADICTION_CLASS:
                confusion_matrix_not_coherent.update([ (target, predicted)  ])
            else:
                confusion_matrix_coherent.update([ (target, predicted)  ])

        if predicted == target:
            correct += 1

    if not include_reverse:
        return correct / float(len(eval_set)), cost, confusion_matrix
    
    if not qualitative:
        return correct / float(len(eval_set)), cost, confusion_matrix, \
               logic_regularizer.validate_inference_rule(hypotheses, reversed), \
               logic_regularizer.validate_contradiction_rule(hypotheses, reversed), \
               logic_regularizer.validate_neutral_rule(hypotheses, reversed), \
               confusion_matrix_coherent, confusion_matrix_not_coherent, inversion_matrix

    def get_worst_logical(loss_fn, original, inverse, sess):
        loss_vals = sess.run(loss_fn(original, inverse))
        idxs = np.argsort(loss_vals)[::-1][:5]
        return idxs, original[idxs, :], inverse[idxs, :], loss_vals[idxs]

    def get_worst_cases(proba, class_idxs, class_id):
        idxs = np.argsort(proba[class_idxs, class_id])[:5]
        idxs = [ class_idxs[i] for i in idxs]
        return idxs, proba[idxs, :]

    with tf.Session() as sess:
        worst_logical_contradiction = get_worst_logical(logic_regularizer.semantic_contradiction, probs, inverse_probs, sess)
        worst_logical_neutral = get_worst_logical(logic_regularizer.semantic_neutral, probs, inverse_probs, sess)
        worst_logical_inference = get_worst_logical(logic_regularizer.semantic_inference, probs, inverse_probs, sess)

    worst_contradiction = get_worst_cases(probs, contradiction_idx, CONTRADICTION_CLASS)
    worst_neutral = get_worst_cases(probs, neutral_idx, NEUTRAL_CLASS)
    worst_inference = get_worst_cases(probs, inference_idx, INFERENCE_CLASS)

    return correct / float(len(eval_set)), cost, confusion_matrix, \
               logic_regularizer.validate_inference_rule(hypotheses, reversed), \
               logic_regularizer.validate_contradiction_rule(hypotheses, reversed), \
               logic_regularizer.validate_neutral_rule(hypotheses, reversed), \
               confusion_matrix_coherent, confusion_matrix_not_coherent, inversion_matrix, \
               (worst_contradiction, worst_neutral, worst_inference), \
               (worst_logical_contradiction, worst_logical_neutral, worst_logical_inference)


def evaluate_classifier_genre(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model by genre, evaluated on a chosen dataset. It returns a dictionary of accuracies by genre and cost for the full evaluation dataset.
    
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost = classifier(eval_set)
    correct = dict((genre,0) for genre in set(genres))
    count = dict((genre,0) for genre in set(genres))
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        genre = genres[i]
        if hypothesis == eval_set[i]['label']:
            correct[genre] += 1.
        count[genre] += 1.

        if genre != eval_set[i]['genre']:
            print('welp!')

    accuracy = {k: correct[k]/count[k] for k in correct}

    return accuracy, cost

def evaluate_classifier_bylength(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model by genre, evaluated on a chosen dataset. It returns a dictionary of accuracies by genre and cost for the full evaluation dataset.
    
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost = classifier(eval_set)
    correct = dict((genre,0) for genre in set(genres))
    count = dict((genre,0) for genre in set(genres))
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        genre = genres[i]
        if hypothesis == eval_set[i]['label']:
            correct[genre] += 1.
        count[genre] += 1.

        if genre != eval_set[i]['genre']:
            print('welp!')

    accuracy = {k: correct[k]/count[k] for k in correct}

    return accuracy, cost

def evaluate_final(restore, classifier, eval_sets, batch_size):
    """
    Function to get percentage accuracy of the model, evaluated on a set of chosen datasets.
    
    restore: a function to restore a stored checkpoint
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    restore(best=True)
    percentages = []
    length_results = []
    for eval_set in eval_sets:
        bylength_prem = {}
        bylength_hyp = {}
        genres, hypotheses, cost = classifier(eval_set)
        correct = 0
        cost = cost / batch_size
        full_batch = int(len(eval_set) / batch_size) * batch_size

        for i in range(full_batch):
            hypothesis = hypotheses[i]
            
            length_1 = len(eval_set[i]['sentence1'].split())
            length_2 = len(eval_set[i]['sentence2'].split())
            if length_1 not in bylength_prem.keys():
                bylength_prem[length_1] = [0,0]
            if length_2 not in bylength_hyp.keys():
                bylength_hyp[length_2] = [0,0]

            bylength_prem[length_1][1] += 1
            bylength_hyp[length_2][1] += 1

            if hypothesis == eval_set[i]['label']:
                correct += 1  
                bylength_prem[length_1][0] += 1
                bylength_hyp[length_2][0] += 1    
        percentages.append(correct / float(len(eval_set)))  
        length_results.append((bylength_prem, bylength_hyp))
    return percentages, length_results


def predictions_kaggle(classifier, eval_set, name):
    """
    Get comma-separated CSV of predictions.
    Output file has two columns: pairID, prediction
    """
    INVERSE_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
    }

    _, hypotheses, _ = classifier(eval_set)
    predictions = []
    
    for i in range(len(eval_set)):
        hypothesis = hypotheses[i]
        prediction = INVERSE_MAP[hypothesis]
        pairID = eval_set[i]["pairID"]  
        predictions.append((pairID, prediction))

    #predictions = sorted(predictions, key=lambda x: int(x[0]))

    f = open( name + '_predictions.csv', 'wb')
    w = csv.writer(f, delimiter = ',')
    w.writerow(['pairID','gold_label'])
    for example in predictions:
        w.writerow(example)
    f.close()
