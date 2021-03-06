import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn

from layer import attention, dice, AUGRU
from utils import sequence_mask

from afm import AFM

class Base(tf.keras.Model):
    def __init__(self, user_count, item_count, cate_count, cate_list,
                       user_dim, item_dim, cate_dim,
                       dim_layers):
        super(Base, self).__init__()
        self.item_dim = item_dim
        self.cate_dim = cate_dim

        self.user_emb = nn.Embedding(user_count, user_dim)
        self.item_emb = nn.Embedding(item_count, item_dim)
        self.cate_emb = nn.Embedding(cate_count, cate_dim)
        self.item_bias= tf.Variable(tf.zeros([item_count]), trainable=True)
        self.cate_list = cate_list

        self.hist_bn = nn.BatchNormalization()

        self.hist_fc = nn.Dense(item_dim+cate_dim)

        self.fc = tf.keras.Sequential()
        self.fc.add(nn.BatchNormalization())
        for dim_layer in dim_layers[:-1]:
            self.fc.add(nn.Dense(dim_layer, activation='sigmoid'))
        self.fc.add(nn.Dense(dim_layers[-1], activation=None))

    def get_emb(self, user, item, history):
        user_emb = self.user_emb(user)

        item_emb = self.item_emb(item)
        item_cate_emb = self.cate_emb(tf.gather(self.cate_list, item))
        item_join_emb = tf.concat([item_emb, item_cate_emb], -1)
        item_bias= tf.gather(self.item_bias, item)

        hist_emb = self.item_emb(history)
        hist_cate_emb = self.cate_emb(tf.gather(self.cate_list, history))
        hist_join_emb = tf.concat([hist_emb, hist_cate_emb], -1)

        return user_emb, item_join_emb, item_bias, hist_join_emb

    def call(self, user, item, history, length):
        user_emb, item_join_emb, item_bias, hist_join_emb = self.get_emb(user, item, history)

        hist_mask = tf.sequence_mask(length, max(length), dtype=tf.float32)
        hist_mask = tf.tile(tf.expand_dims(hist_mask, -1), (1,1,self.item_dim+self.cate_dim))
        hist_join_emb = tf.math.multiply(hist_join_emb, hist_mask)
        hist_join_emb = tf.reduce_sum(hist_join_emb, 1)
        hist_join_emb = tf.math.divide(hist_join_emb, tf.cast(tf.tile(tf.expand_dims(length, -1),
                                                      [1,self.item_dim+self.cate_dim]), tf.float32))

        hist_hid_emb = self.hist_fc(self.hist_bn(hist_join_emb))
        join_emb = tf.concat([user_emb, item_join_emb, hist_hid_emb], -1)

        output = tf.squeeze(self.fc(join_emb)) + item_bias
        logit = tf.keras.activations.sigmoid(output)

        return output, logit


class DIN(Base):
    def __init__(self, user_count, item_count, cate_count, cate_list,
                       user_dim, item_dim, cate_dim,
                       dim_layers):
        super(DIN, self).__init__(user_count, item_count, cate_count, cate_list,
                                  user_dim, item_dim, cate_dim,
                                  dim_layers)

        self.afm = AFM('att')

        self.hist_at = attention(item_dim+cate_dim, dim_layers)

        self.fc = tf.keras.Sequential()
        self.fc.add(nn.BatchNormalization())

        for dim_layer in dim_layers[:-1]:
            self.fc.add(nn.Dense(dim_layer, activation=None))
            self.fc.add(dice(dim_layer))
        self.fc.add(nn.Dense(dim_layers[-1], activation=None))

        self.method_idx = None
        return

    def set_method(self, method_idx):
        self.method_idx = method_idx[0]
        self.hist_at.set_method(method_idx[1])
        return True

    def source_method(self, user, item, history, length):
        '''
        method 0: baseline method
        (32, 128) (32, 128) (32,)
        hist_join_emb (32, 33, 128),(32, 266, 128) ???????????????????????????
        '''
        user_emb, item_join_emb, item_bias, hist_join_emb = self.get_emb(user, item, history)
        hist_attn_emb = self.hist_at(item_join_emb, hist_join_emb, length) #(32, 128)
        hist_attn_emb = self.hist_fc(self.hist_bn(hist_attn_emb)) #(32, 128)
        join_emb = tf.concat([user_emb, item_join_emb, hist_attn_emb], -1) #(32, 384)
        output = tf.squeeze(self.fc(join_emb)) + item_bias#(32,)
        logit = tf.keras.activations.sigmoid(output)

        return output, logit

    def add_afm_to_output(self, user, item, history, length):
        '''
        method 1: merge afm result with hist_attn_emb
        '''
        user_emb, item_join_emb, item_bias, hist_join_emb = self.get_emb(user, item, history)
        afm_filter = self.afm(hist_join_emb) #(32, 1)
        hist_attn_emb = self.hist_at(item_join_emb, hist_join_emb, length) #(32, 128)
        hist_attn_emb = self.hist_fc(self.hist_bn(hist_attn_emb)) #(32, 128)
        join_emb = tf.concat([user_emb, item_join_emb, hist_attn_emb], -1) #(32, 384)
        output = tf.squeeze(self.fc(join_emb)) + item_bias + afm_filter #(32,)
        logit = tf.keras.activations.sigmoid(output)

        return output, logit

    def add_afm_to_attention_output(self, user, item, history, length):
        '''
        method 2: merge afm result with hist_attn_emb
        '''
        user_emb, item_join_emb, item_bias, hist_join_emb = self.get_emb(user, item, history)
        afm_filter = self.afm(hist_join_emb) #(32, 1)
        hist_attn_emb = self.hist_at(item_join_emb, hist_join_emb, length) #(32, 128)
        hist_attn_emb_concat_afm = tf.concat([hist_attn_emb, afm_filter], -1) #(32, 129)
        hist_attn_emb = self.hist_fc(self.hist_bn(hist_attn_emb_concat_afm)) #(32, 128)
        join_emb = tf.concat([user_emb, item_join_emb, hist_attn_emb], -1) #(32, 384)
        output = tf.squeeze(self.fc(join_emb)) + item_bias #(32,)
        logit = tf.keras.activations.sigmoid(output)

        return output, logit

    def add_afm_to_attention_output_with_candidate(self, user, item, history, length):
        '''
        method 3: merge afm result with hist_attn_emb, add candidate to afm
        '''
        user_emb, item_join_emb, item_bias, hist_join_emb = self.get_emb(user, item, history)
        item_join_emb_updim = np.array(item_join_emb).reshape([item_join_emb.shape[0], 1, item_join_emb.shape[1]])
        hist_join_emb_concat_candidate = tf.concat([item_join_emb_updim, hist_join_emb], 1) #(32, *, 128)
        afm_filter = self.afm(hist_join_emb_concat_candidate) #(32, 1)
        hist_attn_emb = self.hist_at(item_join_emb, hist_join_emb, length) #(32, 128)
        hist_attn_emb_concat_afm = tf.concat([hist_attn_emb, afm_filter], -1) #(32, 129)
        hist_attn_emb = self.hist_fc(self.hist_bn(hist_attn_emb_concat_afm)) #(32, 128)
        join_emb = tf.concat([user_emb, item_join_emb, hist_attn_emb], -1) #(32, 384)
        output = tf.squeeze(self.fc(join_emb)) + item_bias #(32,)
        logit = tf.keras.activations.sigmoid(output)

        return output, logit

    def call(self, user, item, history, length):
        if self.method_idx == 0:
            return self.source_method(user, item, history, length)

        if self.method_idx == 1:
            return self.add_afm_to_output(user, item, history, length)

        if self.method_idx == 2:
            return self.add_afm_to_attention_output(user, item, history, length)

        if self.method_idx == 3:
            return self.add_afm_to_attention_output_with_candidate(user, item, history, length)

        print("method out of range! program will exit...")
        exit()
        return None, None


class DIEN(Base):
    def __init__(self, user_count, item_count, cate_count, cate_list,
                       user_dim, item_dim, cate_dim,
                       dim_layers):
        super(DIEN, self).__init__(user_count, item_count, cate_count, cate_list,
                                   user_dim, item_dim, cate_dim,
                                   dim_layers)

        self.hist_gru = nn.GRU(item_dim+cate_dim, return_sequences=True)
        self.hist_augru = AUGRU(item_dim+cate_dim)

    def call(self, user, item, history, length):
        user_emb, item_join_emb, item_bias, hist_join_emb = self.get_emb(user, item, history)

        hist_gru_emb = self.hist_gru(hist_join_emb)
        hist_mask = tf.sequence_mask(length, max(length), dtype=tf.bool)
        hist_mask = tf.tile(tf.expand_dims(hist_mask, -1), (1,1,self.item_dim+self.cate_dim))
        hist_attn = tf.nn.softmax(tf.matmul(tf.expand_dims(item_join_emb, 1), hist_gru_emb, transpose_b=True))

        hist_hid_emb = tf.zeros_like(hist_gru_emb[:,0,:])
        for in_emb, in_att in zip(tf.transpose(hist_gru_emb, [1,0,2]),
                                  tf.transpose(hist_attn, [2,0,1])):
            hist_hid_emb = self.hist_augru(in_emb, hist_hid_emb, in_att)

        join_emb = tf.concat([user_emb, item_join_emb, hist_hid_emb], -1)

        output = tf.squeeze(self.fc(join_emb)) + item_bias
        logit = tf.keras.activations.sigmoid(output)

        return output, logit
