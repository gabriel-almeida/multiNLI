import tensorflow as tf


def inference_rule(prob_ab, prob_ba):
    inference_value1 = prob_ab[:, 2]
    contradition_value2 = prob_ba[:, 0]
    return 1.0 - tf.maximum(inference_value1 + contradition_value2 - 1.0, 0.0)


def contradiction_rule(prob_ab, prob_ba):
    contradition_value1 = prob_ab[:, 0]
    contradition_value2 = prob_ba[:, 0]
    a = tf.minimum(1.0 - contradition_value1 + contradition_value2, 1.0)
    b = tf.minimum(1.0 - contradition_value2 + contradition_value1, 1.0)
    return tf.maximum(a + b - 1.0, 0.0)


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
