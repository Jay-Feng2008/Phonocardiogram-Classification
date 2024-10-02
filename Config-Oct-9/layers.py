import tensorflow as tf
from tensorflow import keras
import numpy as np

class ProbSparseAttention(keras.layers.Layer):
    def __init__(self, factor=5):
        super(ProbSparseAttention, self).__init__()
        self.factor = factor

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L, E = K.shape # [B, H, L, D]
        _, _, S, _ = Q.shape # [B, H, L, D]

        # calculate the sampled Q_K
        K_expand = tf.broadcast_to(tf.expand_dims(K, -3), (B, H, S, L, E)) # [B, H, L, L, D]

        indx_q_seq = tf.random.uniform((S,), maxval=L, dtype=tf.int32)
        indx_k_seq = tf.random.uniform((sample_k,), maxval=L, dtype=tf.int32)

        K_sample = tf.gather(K_expand, tf.range(S), axis=2) # [B, H, L, L, D]

        K_sample = tf.gather(K_sample, indx_q_seq, axis=2) # [B, H, L~, L, D]
        K_sample = tf.gather(K_sample, indx_k_seq, axis=3) # [B, H, L~, sample_k, D]

        Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, -2), tf.einsum("...ij->...ji", K_sample))) # [B, H, L~, D]
        # find the Top_k query with sparisty measurement
        M = tf.math.reduce_max(Q_K_sample, axis=-1) - tf.raw_ops.Div(x=tf.reduce_sum(Q_K_sample, axis=-1), y=L)
        M_top = tf.math.top_k(M, n_top, sorted=False)[1]
        batch_indexes = tf.tile(tf.range(Q.shape[0])[:, tf.newaxis, tf.newaxis], (1, Q.shape[1], n_top))
        head_indexes = tf.tile(tf.range(Q.shape[1])[tf.newaxis, :, tf.newaxis], (Q.shape[0], 1, n_top))

        idx = tf.stack(values=[batch_indexes, head_indexes, M_top], axis=-1)

        # use the reduced Q to calculate Q_K
        Q_reduce = tf.gather_nd(Q, idx) # [B, H, n_top, D]

        Q_K = tf.matmul(Q_reduce, tf.transpose(K, [0, 1, 3, 2])) # [B, H, n_top, L]

        return Q_K, idx

    def call(self, x):
        Q, K, V = x
        B, L, H, D = Q.shape
        Q = tf.reshape(Q, (B, H, L, -1))
        K = tf.reshape(K, (B, H, L, -1))
        V = tf.reshape(V, (B, H, L, -1))

        U = self.factor * np.ceil(np.log(L)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()
        # u = L  # Didn't work!! (testing)accuracy/f1 didn't improve. training converge as normal. sampling acts as the dropouts in canonical transformer.

        scores_top, idx = self._prob_QK(Q, K, u, U)
        V_sum = tf.reduce_sum(V, -2)
        context = tf.identity(tf.broadcast_to(tf.expand_dims(V_sum, -2), [B, H, L, V_sum.shape[-1]])) # [B, H, L, D]

        # update the context with selected top_k queries
        attn = tf.keras.activations.softmax(scores_top, axis=-1)
        context = tf.tensor_scatter_nd_update(context, idx, tf.matmul(attn, V))

        return tf.convert_to_tensor(context)


class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = ProbSparseAttention()
        self.d_model = d_model
        self.num_heads = num_heads

        self.query_projection = tf.keras.layers.Dense(d_model)
        self.key_projection = tf.keras.layers.Dense(d_model)
        self.value_projection = tf.keras.layers.Dense(d_model)
        self.out_projection = tf.keras.layers.Dense(d_model)
        self.num_heads = num_heads

    def call(self, x):
        B, L, _ = x.shape  # [B, L, D] --projection--> [B, L, proj_dim]
        H = self.num_heads

        Q = tf.reshape(self.query_projection(x), (B, L, H, -1))
        K = tf.reshape(self.key_projection(x), (B, L, H, -1))
        V = tf.reshape(self.value_projection(x), (B, L, H, -1))

        out = tf.reshape(self.attention([Q, K, V]), (B, L, -1))

        return self.out_projection(out) # [B, L, D]


class ConvLayer(keras.layers.Layer):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = tf.keras.layers.Conv1D(
                                  filters=c_in,
                                  kernel_size=3,
                                  padding='causal')
        self.activation = tf.keras.layers.ELU()
        self.maxPool = tf.keras.layers.MaxPool1D(pool_size=3, strides=2)

    def call(self, x, **kargs):
        x = self.downConv(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x

class FeedForward(keras.layers.Layer):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.dense1 = keras.layers.Dense(d_model, activation='gelu')
        self.dense2 = keras.layers.Dense(d_model)
        self.distill = ConvLayer(d_model)
        self.norm = keras.layers.LayerNormalization()

    def call(self, x):
        x = self.distill(x)
        x = keras.layers.Add()([self.dense1(x), x])
        x = self.dense2(x)
        x = self.norm(x)
        return x

def positional_encoding(length, depth):
        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)    

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.embd = keras.layers.Conv1D(filters=d_model, kernel_size=1)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embd(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
