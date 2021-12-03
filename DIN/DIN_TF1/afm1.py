import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import itertools


def afm(sequence, mode):
    row = []
    col = []

    # TODO sequence [b, t, h]
    for r, c in itertools.combinations(range(100), 2):
        row.append(r)
        col.append(c)
    p = tf.gather(sequence, row, axis=1)  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
    q = tf.gather(sequence, col, axis=1)  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
    bi_interaction = p * q  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
    # TODO (None, 741, 32)
    # print(bi_interaction.shape)

    # mode
    if mode == 'max':
        # MaxPooling Layer
        x = tf.reduce_sum(bi_interaction, axis=1)   # (None, k)
    elif mode == 'avg':
        # AvgPooling Layer
        x = tf.reduce_mean(bi_interaction, axis=1)  # (None, k)
    else:
        # Attention Layer
        x = attention(bi_interaction,sequence.get_shape().as_list()[-1])  # (None, k)
    # Output Layer
    outputs = tf.nn.sigmoid(tf.layers.dense(inputs=x, units=1, activation=None))

    return outputs


def attention(bi_interaction, att_vec):
    # TODO (None, (len(sparse) * len(sparse) - 1) / 2, t)
    a = tf.layers.dense(inputs=bi_interaction, units=att_vec, activation='relu')
    # TODO (None, 741, 8)
    # print(a.shape)

    # TODO (None, (len(sparse) * len(sparse) - 1) / 2, 1)
    a = tf.layers.dense(inputs=a, units=1, activation=None)
    # TODO (None, 741, 1)
    # print(a.shape)

    # TODO (None, (len(sparse) * len(sparse) - 1) / 2, 1)
    a_score = tf.nn.softmax(a, axis=1)
    # TODO (None, embed_dim)
    # outputs = bi_interaction * a_score
    outputs = tf.reduce_sum(bi_interaction * a_score, axis=1)

    return outputs