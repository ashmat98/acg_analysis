import pickle
import numpy as np
import invrep_supervised
from sklearn.utils import class_weight
import os
import tensorflow as tf

import sys; sys.path.append("..")
from utils import one_hot, batch_generator
import data_process_tools

print("Reading data...")
data_file = "../ecg_data_2/samples/data.train-val.163patients.156544+78278samples.800points.19-09-11.18:52:43.pkl"
with open(data_file, "rb") as f:
    train, qval, val = data_process_tools.label(
        data_process_tools.split(pickle.load(f), 
        train_portion=0.6, val_portion=0.4, seed=202))

# select channels
for obj in (train, qval, val):
    obj.x = obj.x[:, :, 2:]

# set one-hot inputs
for obj in (train, qval, val):
    for attr in ("y", "p"):
        setattr(obj, attr, one_hot(getattr(obj, attr)))

class_weights=class_weight.compute_class_weight("balanced", [0,1], train.y.argmax(axis=-1))
print("Class weights:", class_weights)

train.batches = batch_generator(train.x, train.y, train.p, batch_size=100, infinite=True)
qval.batches = batch_generator(qval.x, qval.y, qval.p, batch_size=100, infinite=True)
val.batches = batch_generator(val.x, val.y, val.p, batch_size=100, infinite=True)


print("Building model...")
drop_rate = 0.1

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
            h = tf.layers.dense(h, 100, activation="relu")
            h = tf.layers.dropout(h, drop_rate)
            y_hat = tf.layers.dense(h, output_shape, activation="linear")
        return y_hat


epoches = 35
adam_lr = 0.001
verbose = True

def search_for(lambda_param, beta_param):
    tf.reset_default_graph()
    model = Model_invrep(input_shape=train.x.shape[-2:], latent_dim=100, 
                n_labels=train.y.shape[1], n_confounds=train.p.shape[1],
                class_weights=class_weights, drop_rate=drop_rate,
                lambda_param=lambda_param, beta_param=beta_param,
                learning_rate=adam_lr)
    for epoch in range(epoches):
        eval_steps = 200

        model.train_invrep(train.batches, steps=150, verbose=verbose)
        model.evaluate(qval.batches, steps=eval_steps, label="qeval", verbose=verbose)
        model.evaluate(val.batches, steps=eval_steps, label="eval", verbose=verbose)
    
        # model.train_adversarial(train_batches, epochs=5, steps=50, verbose=verbose)
        # model.eval_adversarial(qval_batches, steps=eval_steps, label="qeval", verbose=verbose)
        # model.eval_adversarial(val_batches, steps=eval_steps, label="eval", skip_c=True, verbose=verbose)
    
    model.save("./runs", prefix="run_multiple")

# search_for(0.003, 0.003)
# exit(0)

print("Starting gridsearch...")

lambdas = list(np.logspace(1, -5, 15).round(8)) + [0, 100]
betas = list(np.logspace(1, -5, 15).round(8)) + [0, 100]
values = []
for l in lambdas:
    for b in betas:
        values.append({"lambda_param":l, "beta_param":b})
np.random.seed(42)
np.random.shuffle(values)

for _ in range(10):
    for x in values:
        search_for(**x)