import tensorflow.keras.backend as K
from tensorflow.keras.losses import MSE, KLD, MAE
from module import *
import tensorflow as tf
import numpy as np
from sklearn import metrics
from loss import ZINB, pairwise_loss, cal_dist
import tensorflow_probability as tfp
from layers import *
from sklearn.cluster import KMeans


def update_centers(z, q_out):
    # number of clusters
    n_clusters = q_out.shape[1]
    # K-means computes a new clustering center of mass
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(z.numpy())

    return kmeans.cluster_centers_


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def generate_batches(data, batch_size):
    num_samples = data.shape[0]
    num_batches = num_samples // batch_size
    if num_samples % batch_size != 0:
        num_batches += 1
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        yield data[start_idx:end_idx]

class scG(tf.keras.Model):
    def __init__(self, raw_X, X, size_factor, model_pth, G_adj, G_adj_n, L_adj, L_adj_n, hidden_dim=256,
                 latent_dim=128, dec_dim=None, adj_dim=32):
        super(scG, self).__init__()

        if dec_dim is None:
            self.dec_dim = [128, 256, 512]
        else:
            self.dec_dim = dec_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.raw_X = raw_X
        self.X = X
        self.size_factors = tf.convert_to_tensor(size_factor, dtype="float32")
        self.model_pth = model_pth


        self.G_adj = np.float32(G_adj)
        self.G_adj_n = tfp.math.dense_to_sparse(np.float32(G_adj_n))
        self.L_adj = np.float32(L_adj)
        self.L_adj_n = tfp.math.dense_to_sparse(np.float32(L_adj_n))


        self.n_sample = X.shape[0]        # cell
        self.in_dim = X.shape[1]          # gene
        self.raw_dim = raw_X.shape[1]
        self.adj_dim = adj_dim
        self.sparse = True


        self.encoder = EncoderG(X, G_adj_n, L_adj_n, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        self.decoderX = DecoderX(self.latent_dim, self.raw_dim, self.dec_dim)
        self.decoderA = DecoderA(self.adj_dim)

        self.cluster_layer = ClusteringLayer(n_clusters=4, name='clustering')

        self.encoder.build(input_shape=(None, self.in_dim))
        self.decoderA.build(input_shape=(None, self.latent_dim))
        self.decoderX.build(input_shape=(None, self.latent_dim))
        self.cluster_layer.build(input_shape=(None, self.latent_dim))


    def pre_train(self, epochs=800, info_step=20, lr=0.001, W_a=0.0001, W_x=1.0, W_r=0.001, patience=20):
        W_a = tf.constant(W_a, dtype=tf.float32)
        W_x = tf.constant(W_x, dtype=tf.float32)
        W_r = tf.constant(W_r, dtype=tf.float32)


        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=epochs, decay_rate=0.99)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        import math
        best_loss = math.inf
        patience_count = 0

        # Pretraining
        for epoch in range(1, epochs + 1):
            with tf.GradientTape(persistent=True) as tape:
                # X reconstructed loss
                z = self.encoder(self.X, training=True)
                pi, disp, mean = self.decoderX(z, training=True)
                zinb = ZINB(pi=pi, theta=disp, scale_factor=self.size_factors, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.raw_X, mean, mean=True)

                # A reconstructed loss
                A_out = self.decoderA(z)
                A_rec_loss = tf.reduce_mean(MSE(self.L_adj, A_out)) + tf.reduce_mean(MSE(self.G_adj, A_out))


                reg_loss = tf.reduce_sum(self.encoder.losses) + tf.reduce_sum(self.decoderX.losses) + tf.reduce_sum(
                    self.decoderA.losses)

                loss = W_a * A_rec_loss + W_x * zinb_loss + W_r * reg_loss


            if epoch % info_step == 0:
                print("Epoch", epoch, "total_loss:", loss.numpy())

            # Early stop
            if loss < best_loss:
                best_loss = loss
                patience_count = 0
                try:
                    self.encoder.save_weights(self.model_pth + "pretrain_encoder.h5")
                    self.decoderA.save_weights(self.model_pth + "pretrain_decoderA.h5")
                    self.decoderX.save_weights(self.model_pth + "pretrain_decoderX.h5")
                    print("Weights saved successfully.")
                except Exception as e:
                    print(f"Error saving weights: {e}")
            else:
                patience_count += 1
                if patience_count == patience:
                    print(f"Early stopping at epoch {epoch} with best_loss {best_loss}")
                    break

            vars = self.trainable_weights
            grads = tape.gradient(loss, vars)
            grads, global_norm = tf.clip_by_global_norm(grads, 5)
            optimizer.apply_gradients(zip(grads, vars))

        print("Pre_train Finish!")

    def train(self, y, epochs=300, lr=0.0009, W_a=0.01, W_x=1.0, W_c=1.0, info_step=1, n_update=20,
              centers=None):

        W_a = tf.constant(W_a, dtype=tf.float32)
        W_x = tf.constant(W_x, dtype=tf.float32)
        W_c = tf.constant(W_c, dtype=tf.float32)

        self.cluster_layer.clusters.assign(tf.convert_to_tensor(centers, dtype=tf.float32))

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=epochs, decay_rate=0.99)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        import math
        best_loss = math.inf
        y_last_pred = []
        loss_count = 0

        for epoch in range(0, epochs):
            # Updating the clustering center
            if epoch % n_update == 0:
                z = self.encoder(self.X)
                q = self.cluster_layer(z)
                p = self.target_distribution(q)
                y_pred = q.numpy().argmax(1)
                if epoch == 0:
                    y_last_pred = y_pred
                acc = np.round(cluster_acc(y, y_pred), 5)
                y_pred = np.array(y_pred)
                nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                nmi_arithmetic = np.round(metrics.normalized_mutual_info_score(y, y_pred, average_method='arithmetic'),
                                          5)
                ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                if epoch == 0 or len(set(y_pred)) == 1:
                    sc = 0.0
                else:
                    sc = np.round(metrics.silhouette_score(z, y_pred), 5)
                print('epoch=%d, ACC= %.4f, NMI= %.4f, nmi_arithmetic= %.4f, ARI= %.4f, SC=%.4f' % (epoch, acc, nmi, nmi_arithmetic, ari, sc))

                # Recalculating Cluster Centers
                new_centers = np.zeros((self.cluster_layer.n_clusters, z.shape[1]))
                for i in range(self.cluster_layer.n_clusters):
                    indices = np.where(y_pred == i)[0]
                    if len(indices) > 0:
                        new_centers[i, :] = tf.reduce_mean(tf.gather(z, indices), axis=0)
                self.cluster_layer.clusters.assign(tf.convert_to_tensor(new_centers, dtype=tf.float32))

            with tf.GradientTape(persistent=True) as tape:
                z = self.encoder(self.X, training=True)
                q_out = self.cluster_layer(z)
                cluster_loss = tf.reduce_mean(KLD(q_out, p))
                y_pred = q_out.numpy().argmax(1)
                delta_y = 1 - cluster_acc(y_last_pred, y_pred)
                y_last_pred = y_pred

                pi, disp, mean = self.decoderX(z, training=True)
                zinb = ZINB(pi, theta=disp, scale_factor=self.size_factors, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.raw_X, mean, mean=True)

                A_out = self.decoderA(z)
                A_rec_loss = tf.reduce_mean(MSE(self.L_adj, A_out)) + tf.reduce_mean(MSE(self.G_adj, A_out))

                dist1, dist2 = cal_dist(z, self.cluster_layer.clusters)
                soft_kmeans = tf.reduce_mean(tf.reduce_sum(dist2, axis=1))

                tot_loss = W_a * A_rec_loss + W_x * zinb_loss + W_c * cluster_loss + 0.5 * soft_kmeans

                stop_loss = tot_loss

            if epoch % info_step == 0:
                print("Epoch", epoch, "total_loss:", tot_loss.numpy())

            if stop_loss < best_loss:
                best_loss = stop_loss
                loss_count = 0
                try:
                    self.encoder.save_weights(self.model_pth + "train_encoder.h5")
                    self.decoderA.save_weights(self.model_pth + "train_decoderA.h5")
                    self.decoderX.save_weights(self.model_pth + "train_decoderX.h5")
                    print("Weights1 saved successfully.")
                except Exception as e:
                    print(f"Error saving weights: {e}")
            else:
                loss_count += 1

            if delta_y < 0.001 or loss_count == 10:
                if delta_y < 0.001:
                    try:
                        self.encoder.save_weights(self.model_pth + "train_encoder.h5")
                        self.decoderA.save_weights(self.model_pth + "train_decoderA.h5")
                        self.decoderX.save_weights(self.model_pth + "train_decoderX.h5")
                        print("Weights2 saved successfully.")
                    except Exception as e:
                        print(f"Error saving weights: {e}")
                break

            vars = self.trainable_weights
            grads = tape.gradient(tot_loss, vars)
            grads, global_norm = tf.clip_by_global_norm(grads, 5)
            optimizer.apply_gradients(zip(grads, vars))
    def embedding(self, count):
        embedding = self.encoder(count)
        return np.array(embedding)

    def get_cluster(self):
        z = self.encoder(self.X)
        q = self.cluster_layer(z)
        q = q.numpy()
        y_pred = q.argmax(1)
        return z, y_pred

    def target_distribution(self, q):
        q = q.numpy()
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def load_model(self, mode="pretrain"):
        if mode == "pretrain":
            # save weights
            self.encoder.load_weights(self.model_pth + "pretrain_encoder.h5")
            self.decoderA.load_weights(self.model_pth + "pretrain_decoderA.h5")
            self.decoderX.load_weights(self.model_pth + "pretrain_decoderX.h5")
        elif mode == "train":
            self.encoder.load_weights(self.model_pth + "train_encoder.h5")
            self.decoderA.load_weights(self.model_pth + "train_decoderA.h5")
            self.decoderX.load_weights(self.model_pth + "train_decoderX.h5")


