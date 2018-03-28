import tensorflow as tf


INFERENCE_CLASS = 2
NEUTRAL_CLASS = 1
CONTRADICTION_CLASS = 0

def inference_rule(prob_ab, prob_ba):
    inference_value1 = prob_ab[:, 2]
    contradition_value2 = prob_ba[:, 0]
    return 1.0 - tf.maximum(inference_value1 + contradition_value2 - 1.0, 0.0)


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
    contradiction_prob1 = prob_ab[:, 0]
    contradiction_prob2 = prob_ba[:, 0]
    return -tf.log((1.0 - contradiction_prob1) * (1.0 - contradiction_prob2) +
                   contradiction_prob2 * contradiction_prob2)


def semantic_only_one(probs):
    inference_prob = probs[:, 2]
    neutral_prob = probs[:, 1]
    contradition_prob = probs[:, 0]
    return -tf.log(inference_prob * (1.0 - neutral_prob) * (1.0 - contradition_prob) +
                   (1.0 - inference_prob) * neutral_prob * (1.0 - contradition_prob) +
                   (1.0 - inference_prob) * (1.0 - neutral_prob) * contradition_prob)


def inference_regularization_squared(prob_ab, prob_ba):
    inference_value1 = prob_ab[:, 2]
    contradition_value2 = prob_ba[:, 0]
    return tf.square(tf.maximum(inference_value1 + contradition_value2 - 1.0, 0.0))


def contradiction_rule_v1(prob_ab, prob_ba):
    contradition_value1 = prob_ab[:, 0]
    contradition_value2 = prob_ba[:, 0]
    a = tf.minimum(1.0 - contradition_value1 + contradition_value2, 1.0)
    b = tf.minimum(1.0 - contradition_value2 + contradition_value1, 1.0)
    return tf.maximum(a + b - 1.0, 0.0)


def contradiction_regularization_squared(prob_ab, prob_ba): 
    contradition_prob1 = prob_ab[:, 0]
    contradition_prob2 = prob_ba[:, 0]
    return tf.square(contradition_prob1 -  contradition_prob2 )


def contradiction_rule(prob_ab, prob_ba): 
    contradition_prob1 = prob_ab[:, 0]
    contradition_prob2 = prob_ba[:, 0]
    return 1.0 - tf.abs(contradition_prob1 -  contradition_prob2 )


def calculate_pi(pi_zero, alpha, n_iteration):
    return tf.minimum(pi_zero, 1.0 - tf.pow(alpha, n_iteration))


def q_star(prob_ab, prob_ba, lambda_entailment, lambda_contradition, C_regularizer):
    entail_expoent = lambda_entailment*(1.0 - inference_rule(prob_ab, prob_ba))
    contradict_expoent = lambda_contradition*(1.0 - contradiction_rule(prob_ab, prob_ba))

    return tf.multiply(prob_ab, tf.expand_dims(tf.exp(-C_regularizer*(entail_expoent + contradict_expoent)), dim=1))


def kl_divergence(y_target, y_pred):
    return tf.reduce_mean(tf.multiply(y_target, tf.log(y_pred/y_target)))


def logic_loss(prob_ab, prob_ba):
    inference = inference_rule(prob_ab, prob_ba)
    contradiction = contradiction_rule(prob_ab, prob_ba)
    return tf.exp(1.0 - (inference + contradiction)/2.0) - 1.0


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
