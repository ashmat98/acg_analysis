import numpy as np 
import pickle
import datetime
import os
from utils import get_accuracy, get_f1
import kl_tools
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.1

class Model:

    def __init__(self, input_shape, latent_dim, n_labels, n_confounds, 
        lambda_param=0.0001, beta_param=0.01,
        class_weights=1, drop_rate=0.0,
        learning_rate=0.001, learning_rate_adv=0.001,
        save_step_history=False):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        self.n_confounds = n_confounds
        
        self.lambda_param = lambda_param
        self.beta_param = beta_param

        self.class_weights = tf.constant(class_weights, shape=(n_labels,))

        self.learning_rate = learning_rate
        self.learning_rate_adv = learning_rate_adv

        self._build_graph()
        self._build_loss()
        self._build_adversarial()
        self._build_avd_loss()
        
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        self.step_history = [] if save_step_history else None
        self.epoch_history = []
        self.drop_rate = drop_rate
        self.hyperparams = {
            "lambda_param": self.drop_rate,
            "lambda_param": self.lambda_param,
            "beta_param": self.beta_param
        }

    def reset(self):
        self.session.close()
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

    def __del__(self):
        print("Closing session...")
        self.session.close()

    def train_invrep(self, batch_generator, steps=-1, step_history=False, verbose=True):
        c = dict()
        losses = []
        for i, (batch_x, batch_y, batch_c) in \
            enumerate(batch_generator):
            _, batch_loss, batch_y_like, batch_x_like, batch_kl = \
                self.session.run(
                    (self.train_op, self.total_loss, self.y_likelihood, self.x_likelihood, self.other_div), 
                    feed_dict={self.x_in:batch_x, self.y_in:batch_y, self.c_in:batch_c})
            losses.append([batch_loss, batch_y_like, batch_x_like, batch_kl])
            if i == steps-1:
                break
        c["total_loss"], c["y_likelyhood"], c["x_likelyhood"], c["kl_div"] = np.mean(losses, axis=0)
        if self.step_history is not None:
            self.step_history += losses
        self.epoch_history.append(c)
        if verbose==True:
            print(
                "   Train:invrep\tl: {:0.4f}\tx: {:0.4f}\ty: {:0.4f}\tkl: {:0.4f}"
                .format(c["total_loss"],-c["x_likelyhood"], -c["y_likelyhood"], c["kl_div"]))
        return losses

    def train_adversarial(self, batch_generator, epochs=1, steps=-1, verbose=True):
        """
        Training discriminator networks for evaluating latent variable
        
        Parameters
        ----------
        batch_generator : Data source: returns input, label, confounds with minibatches
        epochs : int, optional
            number of epoches, by default 1
        steps : int, optional
            Number of calls to the generator, by default -1 (untill generator finishes)
        verbose : bool, optional
            prints accuracy and f1 scores, by default True
        
        Returns accuracy and f1 scores
        """
        self.session.run(tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='adversarial')))

        c = {"zc_acc":[],"zc_f1":[],"zy_acc":[],"zy_f1":[]} 
        for _ in range(epochs):
            metrics = []
            for i, (batch_x, batch_y, batch_c) in \
                enumerate(batch_generator):
                batch_z = self.session.run(self.z, 
                    feed_dict={self.x_in:batch_x})
                _,_,acc_zc, f1_zc, acc_zy, f1_zy = self.session.run(
                    (self.train_op_adv_zc, self.train_op_adv_zy,
                        self.accuracy_zc, self.f1_zc,
                        self.accuracy_zy, self.f1_zy ), 
                    feed_dict={self.z_in:batch_z, self.y_in:batch_y, self.c_in:batch_c})
                metrics.append([acc_zc, f1_zc, acc_zy, f1_zy])
                if i == steps-1:
                    break
            metrics = np.mean(metrics, axis=0)
            c["zc_acc"].append(metrics[0])
            c["zc_f1"].append(metrics[1])
            c["zy_acc"].append(metrics[2])
            c["zy_f1"].append(metrics[3])
        
        self.epoch_history[-1].setdefault("adv", dict()).update(c)
        if verbose==True:
            print(
                "   Eval :advers\tzc: {:0.2f}% {:0.4f}\tzy: {:0.2f}% {:0.4f}"
                .format(c["zc_acc"][-1], c["zc_f1"][-1], c["zy_acc"][-1], c["zy_f1"][-1]))
        return c
    
    def eval_adversarial(self,batch_generator, steps=-1, verbose=True):
        """
        Evaluate discriminator networks for evaluating latent variable:
            evaluates scores for z->c and z->y prediction
        Parameters
        ----------
        batch_generator : Data source: returns input, label, confounds with minibatches
        steps : int, optional
            Number of calls to the generator, by default -1 (untill generator finishes)
        verbose : bool, optional
            prints accuracy and f1 scores, by default True
        Returns accuracy and f1 scores
        """
        c = {}
        metrics = []
        for i, (batch_x, batch_y, batch_c) in \
            enumerate(batch_generator):
            batch_z = self.session.run(self.z, 
                feed_dict={self.x_in:batch_x})
            acc_zc, f1_zc, acc_zy, f1_zy = self.session.run(
                (self.accuracy_zc, self.f1_zc, self.accuracy_zy, self.f1_zy ), 
                feed_dict={self.z_in:batch_z, self.y_in:batch_y, self.c_in:batch_c})
            metrics.append([acc_zc, f1_zc, acc_zy, f1_zy])
            if i == steps-1:
                break
        c["zc_acc_eval"], c["zc_f1_eval"], \
            c["zy_acc_eval"],c["zy_f1_eval"] = np.mean(metrics, axis=0)
        
        self.epoch_history[-1].setdefault("adv", dict()).update(c)
        if verbose==True:
            print(
                "   Eval :advers\tzc: {:0.2f}% {:0.4f}\tzy: {:0.2f}% {:0.4f}"
                .format(c["zc_acc_eval"], c["zc_f1_eval"], c["zy_acc_eval"], c["zy_f1_eval"]))
        return c
    
    def eval_on_validation(self,batch_generator, steps=-1, verbose=True):
        """
        Evaluate discriminator networks for evaluating latent variable:
            evaluates scores for z->y prediction (z->c is not available in this case)
        Parameters
        ----------
        batch_generator : Data source: returns input, label, confounds with minibatches
        steps : int, optional
            Number of calls to the generator, by default -1 (untill generator finishes)
        verbose : bool, optional
            prints accuracy and f1 scores, by default True
        Returns accuracy and f1 scores
        """
        c = {}
        metrics = []
        for i, (batch_x, batch_y, _) in \
            enumerate(batch_generator):
            batch_z = self.session.run(self.z, 
                feed_dict={self.x_in:batch_x})
            acc_zy, f1_zy = self.session.run(
                (self.accuracy_zy, self.f1_zy ), 
                feed_dict={self.z_in:batch_z, self.y_in:batch_y})
            metrics.append([acc_zy, f1_zy])
            if i == steps-1:
                break
        c["zy_acc_eval_val"], c["zy_f1_eval_val"] = np.mean(metrics,axis=0)
        self.epoch_history[-1].setdefault("adv", dict()).update(c)
        if verbose==True:
            print(
                "   Eval :advers\t\t\t\tzy: {:0.2f}% {:0.4f}"
                .format(c["zy_acc_eval_val"], c["zy_f1_eval_val"]))
        return c
    

    def _build_graph(self):
        """
        Builds and connects all parts of the model
        """
        with tf.variable_scope("invrep_model"):
            self.x_in = tf.placeholder(tf.float32, shape=(None,) + self.input_shape, 
                name='input_data')
            self.c_in = tf.placeholder(tf.float32, shape=[None, self.n_confounds], name="confounds")
            self.y_in = tf.placeholder(tf.float32, shape=[None, self.n_labels], name='output_label')
            
            self.mu, self.sigma, self.z = \
                self._gaussian_encoder(self.x_in, self.latent_dim)
            self.x_hat = self._generative_decoder(
                self.z, self.c_in, self.input_shape)
            self.y_hat = self._generative_classifier(
                self.z, self.n_labels)

    def _build_adversarial(self):
        """
        Builds discriminator networks.
        """
        with tf.variable_scope("adversarial"):
            self.z_in = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name="z_in")
            # adversarial z->c
            self.adv_zc = self._ffnetwork(
                self.z_in, self.n_confounds, "adv_z_to_c",
                hidden_layers=0, hidden_units=100, 
                final_activation=None)
            # adversarial z->y
            self.adv_zy = self._ffnetwork(
                self.z_in, self.n_labels, "adv_z_to_y",
                hidden_layers=0, hidden_units=100, 
                final_activation=None)

    def _build_loss(self):
        """
        Loss function of the "Invariant representation for supervised learning" model
        - x_likelihood
        - y_likelyhood
        - kl_div
        """
        with tf.name_scope("total_loss"):
            ### Loss y_hat
            weights = tf.gather_nd(self.class_weights, 
                tf.argmax(self.y_in, axis=1)[..., None])
            self.y_likelihood = self.classifier_likelihood(self.y_in, self.y_hat, weights)
            ### x_hat total_loss
            self.x_likelihood = self.reconstuction_likelihood(self.x_in, self.x_hat)
            ### kl_loss
            self.other_div = kl_tools.kl_conditional_and_marg(
                self.mu, self.sigma, self.latent_dim)
            ELBO = self.lambda_param * self.x_likelihood + self.y_likelihood\
                - (self.beta_param + self.lambda_param) * self.other_div
            self.total_loss = -ELBO  # should be minimized

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

    def _build_avd_loss(self):
        """
        build metrics for descriminator networks.
        """
        with tf.name_scope("adversarial_loss"):
            self.loss_adv_zy = tf.losses.softmax_cross_entropy(
                self.y_in, self.adv_zy)
            self.loss_adv_zc = tf.losses.softmax_cross_entropy(
                self.c_in, self.adv_zc)

            self.train_op_adv_zy = tf.train.AdamOptimizer(self.learning_rate_adv).minimize(self.loss_adv_zy)
            self.train_op_adv_zc = tf.train.AdamOptimizer(self.learning_rate_adv).minimize(self.loss_adv_zc)

        with tf.name_scope("metrics"):
            self.accuracy_zy = get_accuracy(self.y_in, self.adv_zy)    
            self.accuracy_zc = get_accuracy(self.c_in, self.adv_zc)

            self.f1_zy = get_f1(self.y_in, self.adv_zy)    
            self.f1_zc = get_f1(self.c_in, self.adv_zc)

    @staticmethod
    def _ffnetwork(input_tensor, output_dim, 
                   name_scope, hidden_layers, 
                   hidden_units=100, final_activation=None):
        """
        Fully connected network
        """
        
        with tf.name_scope(name_scope):
            h = input_tensor
            for _ in range(hidden_layers):
                h = tf.layers.dense(h, hidden_units, activation="relu")
            output_tensor = tf.layers.dense(h, output_dim, 
                activation=final_activation)
        return output_tensor

    
    def save(self, path, prefix="run"): 
        """
        Saves all metrics collected during training-evaluation into pickle file.
        
        Parameters
        ----------
        path : str, path to directory
        prefix : str, optional, file name prefix, by default "run"
            run_<date-time>_l<value>_d<value>.pkl
        """
        run_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(path, 
                prefix + "_" + run_datetime + f"_l{self.lambda_param}_b{self.beta_param}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump((self.epoch_history, self.step_history, self.hyperparams), f)
        print(f"History saved in \"{file_path}\".")

    ################################3
    ### can be changed down here
    
    @staticmethod
    def classifier_likelihood(true_tensor, predicted_tensor, weights):
        """
        classifier likelihood for y/label prediction.
        Parameters
        ----------
        weights : batch weight
        """
        return -tf.losses.softmax_cross_entropy(
                onehot_labels=true_tensor, 
                logits=predicted_tensor, 
                weights=weights)
    
    @staticmethod
    def reconstuction_likelihood(true_tensor, predicted_tensor):
        """
        Reconstruction likelihood for x/input reconstruction.
        """
        return -tf.losses.mean_squared_error(
                labels=true_tensor, 
                predictions=predicted_tensor )

    @staticmethod
    def _gaussian_encoder(x_in, latent_dim):
        """
        Encodes input to latent: x->z
        latent_dim : int, dimension of latent space.
        """
        h = x_in

        h = tf.layers.dense(h, 100, activation=tf.nn.relu)
        h = tf.layers.dense(h, 100, activation=tf.nn.relu)

        mu = tf.layers.dense(h, latent_dim, activation=None)
        sigma = tf.layers.dense(h, latent_dim, activation=tf.nn.softplus)
        # representation
        z = mu + (1e-6 + sigma) * tf.random_normal(
                tf.shape(mu), 0, 1, dtype=tf.float32)
        return mu, sigma, z

    @staticmethod
    def _generative_decoder(z_in, c_in, output_shape):
        """
        decodes latent to input: (z,c)->x.
        output_shape: tuple, dimension of the x/input.
        """
        with tf.name_scope("generative_decoder"):
            h = tf.concat(values=[z_in,c_in], axis=1)
            
            h = tf.layers.dense(h, 100, activation=tf.nn.relu)
            h = tf.layers.dense(h, 100, activation=tf.nn.relu)

            x_decoded = tf.layers.dense(h, output_shape[0], activation=None)
        return x_decoded

    @staticmethod
    def _generative_classifier(z_in, labels):
        """
        Classifier latent to label: z->y.
        labels: int, number of labels to classify.
        """
        with tf.name_scope("generative_classifier"):
            h = z_in

            h = tf.layers.dense(h, 100, activation="relu")
            h = tf.layers.dense(h, 100, activation="relu")

            y_hat = tf.layers.dense(h, labels, activation="linear")
        return y_hat

