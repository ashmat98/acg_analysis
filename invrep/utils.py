import numpy as np
import tensorflow as tf
from keras import backend as K

def one_hot(arr):
    """[summary]
    Returns One-hot encoded version of arr.
    """
    index = np.zeros(np.max(arr)+1, dtype=np.int32)
    uq = np.sort(np.unique(arr))
    index[uq] = np.arange(len(uq))
    enc = np.zeros((len(arr), len(uq)), np.float32)
    enc[range(len(arr)), index[arr]] = 1        
    return enc

def batch_generator(*args, batch_size=100, infinite=False):
    """
    Iterates *args by first dimension by batches of batch size.
    """
    while True:
        perm = np.random.permutation(len(args[0]))
        for start in range(0, len(args[0]), batch_size):
            ids = perm[start:start+batch_size]
            yield (x[ids] for x in args)
        if infinite is False:
            break


def get_accuracy(label_true, label_pred):
        return tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(label_true, axis=1), 
                     tf.argmax(label_pred, axis=1)), tf.float32))*100

def get_recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def get_precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def get_f1(y_true, y_pred):
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def predict_with_batches(sess, inputs, outputs, batch_size):
    start = 0
    results = []
    while True:
        batch_input = {}
        for k, v in inputs.items():
            batch_input[k] = v[start:start+batch_size]
        if batch_input[k].shape[0] ==0:
            break
        results.append(sess.run(outputs, feed_dict=batch_input))
        start += batch_size
    return np.concatenate(results, axis=0)

