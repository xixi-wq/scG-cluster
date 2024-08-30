import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, Input, Lambda
from spektral.layers import TAGConv
from tensorflow.keras.initializers import GlorotUniform
from layers import *
from utils import *
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, Add, Attention, Softmax


class EncoderG(tf.keras.Model):
    def __init__(self, X, G_adj_n, L_adj_n, hidden_dim=256, latent_dim=128):
        super(EncoderG, self).__init__()

        self.X = X
        self.n_sample = X.shape[0]
        self.in_dim = X.shape[1]

        self.G_adj_n = tfp.math.dense_to_sparse(np.float32(G_adj_n))
        self.L_adj_n = tfp.math.dense_to_sparse(np.float32(L_adj_n))

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        initializer = GlorotUniform()
        self.drop = Dropout(0.5)
        self.relu = Activation('relu')

        self.TAG_G_1 = TAGConv(channels=self.hidden_dim, kernel_initializer=initializer, name="TAG_G_1")
        self.bn_G = BatchNormalization()
        self.TAG_G_2 = TAGConv(channels=self.latent_dim, kernel_initializer=initializer, name="TAG_G_2")

        self.TAG_L_1 = TAGConv(channels=self.hidden_dim, kernel_initializer=initializer, name="TAG_L_1")
        self.bn_L = BatchNormalization()
        self.TAG_L_2 = TAGConv(channels=self.latent_dim, kernel_initializer=initializer, name="TAG_L_2")

        self.attention = Attention()
        self.attention_dense = Dense(units=self.latent_dim, activation='relu')
        self.attention_softmax = Softmax(axis=1)

        self.add = Add()

        # Adjustment of dimensions
        self.matching_dense_G = Dense(self.latent_dim, kernel_initializer=initializer)
        self.matching_dense_L = Dense(self.latent_dim, kernel_initializer=initializer)


    def call(self, inputs, training=False, **kwargs):
        x = inputs
        x_h = self.drop(x)

# -----------------------------------------------------------
        G_h = self.TAG_G_1([x_h, self.G_adj_n])
        G_h = self.bn_G(G_h, training=training)
        G_h = self.relu(G_h)
        G_z = self.TAG_G_2([G_h, self.G_adj_n])
        G_h_residual = self.matching_dense_G(G_h)
        G_z = self.add([G_z, G_h_residual])      # Addition, to realize the residual connection


        L_h = self.TAG_L_1([x_h, self.L_adj_n])
        L_h = self.bn_L(L_h, training=training)
        L_h = self.relu(L_h)
        L_z = self.TAG_L_2([L_h, self.L_adj_n])
        L_h_residual = self.matching_dense_L(L_h)
        L_z = self.add([L_z, L_h_residual])

# ----------------------------------------------------------
        z = self.add([G_z, L_z])
# ---------------------------------------------------------

        # G_att = self.attention([G_h, L_h])
        # G_att = self.attention_dense(G_att)
        # G_att = self.attention_softmax(G_att)
        #
        # L_att = self.attention([L_h, G_h])
        # L_att = self.attention_dense(L_att)
        # L_att = self.attention_softmax(L_att)
        #
        # G_z_att = tf.multiply(G_z, G_att)
        # L_z_att = tf.multiply(L_z, L_att)
        #
        # z = self.add([G_z_att, L_z_att])

        return z



# A_out
class DecoderA(tf.keras.Model):
    def __init__(self, adj_dim=32):
        super(DecoderA, self).__init__()

        self.adj_dim = adj_dim

        self.Dense = Dense(units=self.adj_dim, activation=None)
        self.Bilinear = Bilinear()
        self.Lambda = Lambda(lambda z: tf.nn.sigmoid(z))

    def call(self, inputs, **kwargs):

        h = self.Dense(inputs)
        h = self.Bilinear(h)
        dec_out = self.Lambda(h)

        return dec_out

    def get_config(self):
        pass


MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

class DecoderX(tf.keras.Model):
    def __init__(self, latent_dim, raw_dim, dec_dim=None):
        super(DecoderX, self).__init__()
        self.raw_dim = raw_dim

        if dec_dim is None:
            self.dec_dim = [128, 256, 512]
        else:
            self.dec_dim = dec_dim
        self.relu = Activation('relu')

        self.fc1 = Dense(units=self.dec_dim[0])
        self.bn1 = BatchNormalization()
        self.fc2 = Dense(units=self.dec_dim[1])
        self.bn2 = BatchNormalization()
        self.fc3 = Dense(units=self.dec_dim[2])
        self.bn3 = BatchNormalization()

        self.fc_pi = Dense(units=self.raw_dim, activation='sigmoid', kernel_initializer='glorot_uniform', name='pi')
        self.fc_disp = Dense(units=self.raw_dim, activation=DispAct, kernel_initializer='glorot_uniform', name='dispersion')
        self.fc_mean = Dense(units=self.raw_dim, activation=MeanAct, kernel_initializer='glorot_uniform', name='mean')

    def call(self, inputs, training=False, **kwargs):

        x = inputs

        x = self.fc1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x, training=training)
        x = self.relu(x)

        pi = self.fc_pi(x)
        disp = self.fc_disp(x)
        mean = self.fc_mean(x)

        print(pi.shape)

        return pi,disp,mean


