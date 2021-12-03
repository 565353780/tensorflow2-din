import tensorflow as tf
import itertools


class AFM(tf.keras.layers.Layer):
    def __init__(self, mode, att_vector=8, activation='relu', dropout=0.5, embed_reg=1e-6):
        super(AFM, self).__init__()
        self.mode = mode
        if self.mode == 'att':
            self.attention_W = tf.keras.layers.Dense(att_vector, activation=activation, use_bias=True)
            self.attention_dense = tf.keras.layers.Dense(1, activation=None)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        row = []
        col = []

        for r, c in itertools.combinations(range(inputs.shape[1]), 2):
            row.append(r)
            col.append(c)

        p = tf.gather(inputs, row, axis=1)
        q = tf.gather(inputs, col, axis=1)

        bi_interaction = p * q

        if self.mode == "max":
            x = tf.reduce_max(bi_interaction, axis=1)

        elif self.mode == 'avg':
            x = tf.reduce_mean(bi_interaction, axis=1)

        else:
            x = self.attention(bi_interaction)

        outputs = tf.nn.sigmoid(self.dense(x))
        #outputs = self.dense(x)

        return outputs

    def attention(self, bi_interaction):
        a = self.attention_W(bi_interaction)

        a = self.attention_dense(a)

        a_score = tf.nn.softmax(a, 1)

        outputs = tf.reduce_sum(bi_interaction * a_score, 1)

        return outputs
