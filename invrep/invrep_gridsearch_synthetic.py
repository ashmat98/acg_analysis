import pickle
import numpy as np
import invrep_supervised
from sklearn.utils import class_weight
import os
import tensorflow as tf

import sys; sys.path.append("..")
from utils import one_hot, batch_generator
import data_process_tools
from data_process_tools import Data

print("Generating data...")
x = np.random.rand(30000, 2) *2 - 1
r = (x[:, 0]**2 + x[:, 1]**2)
x = x[r<1]

y = one_hot((x[:, 0] >0).astype("int"))
c = x[:, 1, None]

class_weights=[1., 1.]
print("Class weights:", class_weights)

ids = np.prod(x, axis=1)>0
train = Data(x=x[ids], y=y[ids], c=c[ids])
ids = np.prod(x, axis=1)<0
val = Data(x=x[ids], y=y[ids], c=c[ids])

train.batches = batch_generator(train.x, train.y, train.c, batch_size=100, infinite=True)
val.batches = batch_generator(val.x, val.y, val.c, batch_size=100, infinite=True)


print("Building model...")
drop_rate = 0.1
sigma = 0.002
class Model_invrep(invrep_supervised.Model):
    @staticmethod
    def _gaussian_encoder(x_in, latent_dim):
        activation = tf.nn.tanh
        h = tf.layers.flatten(x_in)

        h = tf.layers.dense(h, 10, activation=activation)
        h = tf.layers.dropout(h, drop_rate)
        h = tf.layers.dense(h, 10, activation=activation)
        h = tf.layers.dropout(h, drop_rate)
        h = tf.layers.dense(h, 10, activation=activation)
        h = tf.layers.dropout(h, drop_rate)
        h = tf.layers.dense(h, 10, activation=activation)
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
            activation = tf.nn.tanh

            h = tf.concat(values=[z_in,c_in], axis=1)
            
            h = tf.layers.dense(h, 10, activation=activation)
            h = tf.layers.dropout(h, drop_rate)
            h = tf.layers.dense(h, 10, activation=activation)
            h = tf.layers.dropout(h, drop_rate)
            h = tf.layers.dense(h, 10, activation=activation)
            h = tf.layers.dropout(h, drop_rate)
            h = tf.layers.dense(h, 10, activation=activation)
            h = tf.layers.dropout(h, drop_rate)

            x_decoded = tf.layers.dense(h, output_shape[0], activation=None)
#             x_decoded = tf.reshape(x_decoded, (-1, 800, 6))
        return x_decoded

    @staticmethod
    def _generative_classifier(z_in, output_shape):
        with tf.name_scope("generative_classifier"):
            activation = tf.nn.tanh

            h = z_in
            h = tf.layers.dense(h, 5, activation=activation)
            h = tf.layers.dropout(h, drop_rate)
            h = tf.layers.dense(h, 5, activation=activation)
            h = tf.layers.dropout(h, drop_rate)
            h = tf.layers.dense(h, 5, activation=activation)
            h = tf.layers.dropout(h, drop_rate)
            y_hat = tf.layers.dense(h, output_shape, activation="linear")
        return y_hat
    @staticmethod
    def reconstuction_likelihood(true_tensor, predicted_tensor):
        """
        Reconstruction likelihood for x/input reconstruction.
        """
        return -(2/(2*sigma**2))*tf.losses.mean_squared_error(
                labels=true_tensor, 
                predictions=predicted_tensor )

epoches = 350
adam_lr = 0.0005
verbose = True

def search_for(lambda_param, beta_param):
    tf.reset_default_graph()
    model = Model_invrep(input_shape=train.x.shape[-1:], latent_dim=100, 
                n_labels=train.y.shape[1], n_confounds=train.c.shape[1],
                class_weights=class_weights, drop_rate=drop_rate,
                lambda_param=lambda_param, beta_param=beta_param,
                learning_rate=adam_lr)
    for epoch in range(epoches):
        eval_steps = 200

        model.train_invrep(train.batches, steps=10, verbose=verbose)
        model.evaluate(train.batches, steps=eval_steps, label="qeval", verbose=verbose)
        model.evaluate(val.batches, steps=eval_steps, label="eval", verbose=verbose)
    
        # model.train_adversarial(train_batches, epochs=5, steps=50, verbose=verbose)
        # model.eval_adversarial(qval_batches, steps=eval_steps, label="qeval", verbose=verbose)
        # model.eval_adversarial(val_batches, steps=eval_steps, label="eval", skip_c=True, verbose=verbose)
    
    model.save("./runs", prefix="run_synthetic-sigma-2")

# search_for(0.003, 0.003)
# exit(0)

print("Starting gridsearch...")

lambdas = list(np.logspace(1, -6, 15).round(8)) + [0, 100]
betas = list(np.logspace(1, -6, 15).round(8)) + [0, 100]
values = []
for l in lambdas:
    for b in betas:
        values.append({"lambda_param":l, "beta_param":b})

# np.random.seed(42)

for _ in range(10):
    np.random.shuffle(values)
    for x in values:
        search_for(**x)