import tensorflow as tf
from data_processing import LABEL_MAP

INFERENCE_CLASS = LABEL_MAP["entailment"] 
NEUTRAL_CLASS = LABEL_MAP["neutral"]
CONTRADICTION_CLASS = LABEL_MAP["contradiction"]


def semantic_inference(prob_ab, prob_ba):
    inference_prob1 = prob_ab[:, INFERENCE_CLASS]
    contradition_prob2 = prob_ba[:, CONTRADICTION_CLASS]
    return -tf.log(1.0 - inference_prob1 * contradition_prob2)


def semantic_neutral(prob_ab, prob_ba):
    neutral_prob1 = prob_ab[:, NEUTRAL_CLASS]
    contradition_prob2 = prob_ba[:, CONTRADICTION_CLASS]
    return -tf.log(1.0 - neutral_prob1 * contradition_prob2)


def fuzzy_inference(prob_ab, prob_ba):
    inference_prob1 = prob_ab[:, INFERENCE_CLASS]
    contradiction_prob2 = prob_ba[:, CONTRADICTION_CLASS]
    return -tf.log(1.0 - tf.minimum(inference_prob1, contradiction_prob2))


def fuzzy_neutral(prob_ab, prob_ba):
    neutral_prob1 = prob_ab[:, NEUTRAL_CLASS]
    contradiction_prob2 = prob_ba[:, CONTRADICTION_CLASS]
    return -tf.log(1.0 - tf.minimum(neutral_prob1, contradiction_prob2))


def fuzzy_contradiction(prob_ab, prob_ba):
    contradiction_prob1 = prob_ab[:, CONTRADICTION_CLASS]
    contradiction_prob2 = prob_ba[:, CONTRADICTION_CLASS]
    return -tf.log(tf.maximum(tf.minimum(1.0 - contradiction_prob1, 1.0 - contradiction_prob2), tf.minimum(contradiction_prob1, contradiction_prob2)))


def semantic_contradiction(prob_ab, prob_ba):
    contradiction_prob1 = prob_ab[:, CONTRADICTION_CLASS]
    contradiction_prob2 = prob_ba[:, CONTRADICTION_CLASS]
    return -tf.log((1.0 - contradiction_prob1) * (1.0 - contradiction_prob2) +
                   contradiction_prob2 * contradiction_prob2)


def validate_inference_rule(original_preds, reversed_preds):
    valid = 0.
    total = 0.
    for original, reversed in zip(original_preds, reversed_preds):
        if original == INFERENCE_CLASS:
            total += 1.
            if not reversed == CONTRADICTION_CLASS:
                valid += 1.

    return valid, total


def validate_contradiction_rule(original_preds, reversed_preds):
    valid = 0.
    total = 0.
    for original, reversed in zip(original_preds, reversed_preds):
        if original == CONTRADICTION_CLASS:
            total += 1.
            if reversed == CONTRADICTION_CLASS:
                valid += 1.

    return valid, total


def validate_neutral_rule(original_preds, reversed_preds):
    valid = 0.
    total = 0.
    for original, reversed in zip(original_preds, reversed_preds):
        if original == NEUTRAL_CLASS:
            total += 1.
            if not reversed == CONTRADICTION_CLASS:
                valid += 1.

    return valid, total
