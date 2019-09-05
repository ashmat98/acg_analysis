import pickle
import numpy as np
import invrep_supervised
from sklearn.utils import class_weight
import os
import tensorflow as tf

from utils import one_hot, batch_generator

data_file = "data.train-val.76patients.2000+1000samples.800points.19-09-04.16:54:30.pkl"
with open(os.path.join("samples", data_file), "rb") as f:
    beats_part1, labels_part1, beats_part2, labels_part2 = pickle.load(f)
patients_in_train = 57

train_beats = beats_part1[:patients_in_train].reshape(-1, 800, 6)
train_labels = labels_part1[:patients_in_train].reshape(-1)

qval_beats = beats_part2[:patients_in_train].reshape(-1, 800, 6)
qval_labels = labels_part2[:patients_in_train].reshape(-1)

val_beats = beats_part1[patients_in_train:].reshape(-1, 800, 6)
val_labels = labels_part1[patients_in_train:].reshape(-1)

# substruct mean
for x in [train_beats, qval_beats, val_beats]:
    x[:] = x - x.mean(axis=1, keepdims=True)

patients = (set(x["patient"] for x in train_labels) | 
            set(x["patient"] for x in qval_labels) |
            set(x["patient"] for x in val_labels))
P = {p:i for i,p in enumerate(patients)}
print(len(P) , "patients")

def label_processor_to_patient(label):
    return P[label["patient"]]

def label_processor_to_group(label):
    G = {"ctrls":0, "t1posajneg":1, "t1negajpos":2}
    return G[label["group"]]

def label_processor_to_label(label):
    G = {"ctrls":0, "t1posajneg":1, "t1negajpos":1}
    return G[label["group"]]


train_c = one_hot(np.array([label_processor_to_patient(x) for x in train_labels]))
train_y = one_hot(np.array([label_processor_to_label(x) for x in train_labels]))
train_g = (np.array([label_processor_to_group(x) for x in train_labels]))

qval_c = one_hot(np.array([label_processor_to_patient(x) for x in qval_labels]))
qval_y = one_hot(np.array([label_processor_to_label(x) for x in qval_labels]))
qval_g = (np.array([label_processor_to_group(x) for x in qval_labels]))

val_c = one_hot(np.array([label_processor_to_patient(x) for x in val_labels]))
val_y = one_hot(np.array([label_processor_to_label(x) for x in val_labels]))
val_g = (np.array([label_processor_to_group(x) for x in val_labels]))

class_weights=class_weight.compute_class_weight("balanced", [0,1], train_y.argmax(axis=-1))
print("Class weights:", class_weights)

train_batches = batch_generator(train_beats, train_y, train_c, batch_size=100, infinite=True)
qval_batches = batch_generator(qval_beats, qval_y, qval_c, batch_size=100, infinite=True)
val_batches = batch_generator(val_beats, val_y, val_c, batch_size=100, infinite=True)


drop_rate = 0.15

class Model_invrep(invrep_supervised.Model):
    @staticmethod
    def _gaussian_encoder(x_in, latent_dim):
        h = tf.layers.flatten(x_in)

        h = tf.layers.dense(h, 400, activation=tf.nn.relu)
        h = tf.layers.dropout(h, drop_rate)
        h = tf.layers.dense(h, 400, activation=tf.nn.relu)
        h = tf.layers.dropout(h, drop_rate)
        mu = tf.layers.dense(h, latent_dim, activation=None)
        sigma = tf.layers.dense(h, latent_dim, activation=tf.nn.softplus)
        # representation
        z = mu + (1e-6 + sigma) * tf.random_normal(
                tf.shape(mu), 0, 1, dtype=tf.float32)
        return mu, sigma, z

    @staticmethod
    def _generative_decoder(z_in, c_in, output_shape):
        with tf.name_scope("generative_decoder"):
            h = tf.concat(values=[z_in,c_in], axis=1)
            
            h = tf.layers.dense(h, 400, activation=tf.nn.relu)
            h = tf.layers.dropout(h, drop_rate)
            h = tf.layers.dense(h, 400, activation=tf.nn.relu)
            h = tf.layers.dropout(h, drop_rate)

            x_decoded = tf.layers.dense(h, output_shape[0]*output_shape[1], activation=None)
            x_decoded = tf.reshape(x_decoded, (-1, 800, 6))
        return x_decoded

    @staticmethod
    def _generative_classifier(z_in, output_shape):
        with tf.name_scope("generative_classifier"):
            h = z_in
            h = tf.layers.dense(h, 100, activation="relu")
            h = tf.layers.dropout(h, drop_rate)
            
            y_hat = tf.layers.dense(h, output_shape, activation="linear")
        return y_hat


epoches = 13
verbose = True

def search_for(lambda_param, beta_param):
    tf.reset_default_graph()
    model = Model_invrep(input_shape=(800,6), latent_dim=100, 
                n_labels=train_y.shape[1], n_confounds=train_c.shape[1],
                class_weights=class_weights, drop_rate=drop_rate,
                lambda_param=lambda_param, beta_param=beta_param)
    for epoch in range(epoches):
        if epoch == epoches-1:
            eval_steps = 500
        else:
            eval_steps = 1000

        model.train_invrep(train_batches, steps=200, verbose=verbose)
        model.train_adversarial(train_batches, epochs=10, steps=100, verbose=verbose)
        model.eval_adversarial(qval_batches, steps=eval_steps, verbose=verbose)
        model.eval_on_validation(val_batches, steps=eval_steps)
    model.save("./runs", prefix="run_2")


search_for(0.0001, 0.0001)

exit(0)

lambdas = list(np.logspace(1, -5, 15).round(8)) + [0, 100]
betas = list(np.logspace(1, -5, 15).round(8)) + [0, 100]
values = []
for l in lambdas:
    for b in betas:
        values.append({"lambda_param":l, "beta_param":b})
np.random.seed(42)
np.random.shuffle(values)

for x in values:
    search_for(**x)