# %%

import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import time
import pickle
import random
import tensorflow.compat.v1 as tf
from DIN.DIN_TF1.afm1 import afm

tf.disable_eager_execution()


class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        u, i, y, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2])
            y.append(t[3])
            sl.append(len(t[1]))
        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)

        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1

        return self.i, (u, i, y, hist_i, sl)


class DataInputTest:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        u, i, j, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2][0])
            j.append(t[2][1])
            sl.append(len(t[1]))
        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)

        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1

        return self.i, (u, i, j, hist_i, sl)


class Model(object):

    def __init__(self, user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num):
        self.u = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i = tf.placeholder(tf.int32, [None, ])  # [B]
        self.j = tf.placeholder(tf.int32, [None, ])  # [B]
        self.y = tf.placeholder(tf.float32, [None, ])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float32, [])

        hidden_units = 128

        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
        item_b = tf.get_variable("item_b", [item_count],
                                 initializer=tf.constant_initializer(0.0))
        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])

        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

        ic = tf.gather(cate_list, self.i)

        #tf.nn.embedding_lookup()函数的用法主要是选取一个张量里面索引对应的元素
        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(cate_emb_w, ic),
        ], axis=1)

        i_b = tf.gather(item_b, self.i)

        jc = tf.gather(cate_list, self.j)
        j_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.j),
            tf.nn.embedding_lookup(cate_emb_w, jc),
        ], axis=1)
        j_b = tf.gather(item_b, self.j)

        hc = tf.gather(cate_list, self.hist_i)


        # # ####################Bi-Interaction Layer# ####################
        # item_emb_w = 0.5 * (tf.pow(tf.reduce_sum(tf.nn.embedding_lookup(item_emb_w, self.hist_i), axis=1), 2) -
        #                tf.reduce_sum(tf.pow(tf.nn.embedding_lookup(item_emb_w, self.hist_i), 2), axis=1))  # (None, embed_dim)
        # print(item_emb_w.get_shape())
        # # item_emb_w = tf.tile(item_emb_w, [1,1, 64])
        # # item_emb_w=tf.reshape(item_emb_w, [-1,-1, hidden_units//2], name='hist_bn')
        #
        # cate_emb_w = 0.5 * (tf.pow(tf.reduce_sum(tf.nn.embedding_lookup(cate_emb_w, self.hist_i), axis=1), 2) -
        #                     tf.reduce_sum(tf.pow(tf.nn.embedding_lookup(cate_emb_w, self.hist_i), 2), axis=1))  # (Non
        # # cate_emb_w = tf.tile(cate_emb_w, [1, 1, 64])
        # # cate_emb_w = tf.reshape(cate_emb_w, [-1, -1, hidden_units//2], name='hist_bn')
        # h_emb = tf.concat([
        #     item_emb_w,
        #     cate_emb_w,
        # ], axis=1)
        # h_emb = tf.reshape(h_emb, [-1, 1, hidden_units], name='hist_bn')
        # ####################Bi-Interaction Layer# ####################

        # h_emb 的形状是 [b, t, 128]
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            tf.nn.embedding_lookup(cate_emb_w, hc),
        ], axis=2)

        hist_i = attention(i_emb, h_emb, self.sl)
        # -- attention end ---


        hist_i = tf.layers.batch_normalization(inputs=hist_i)
        hist_i = tf.reshape(hist_i, [-1, hidden_units], name='hist_bn')
        hist_i = tf.layers.dense(hist_i, hidden_units, name='hist_fcn')

        u_emb_i = hist_i

        #(None, 1, 128)
        hist_j = attention(j_emb, h_emb, self.sl)
        # -- attention end ---

        #批量归一化操作  [b, 1, h]
        hist_j = tf.layers.batch_normalization(inputs=hist_j, reuse=True)
        hist_j = tf.reshape(hist_j, [-1, hidden_units], name='hist_bn')
        hist_j = tf.layers.dense(hist_j, hidden_units, name='hist_fcn', reuse=True)

        u_emb_j = hist_j
        # [b, h]
        print("u_emb_i 的维度是：{}".format(u_emb_i.get_shape().as_list()))
        # [b, h]
        print("u_emb_j 的维度是：{}".format(u_emb_j.get_shape().as_list()))
        # [b, h]
        print("i_emb 的维度是：{}".format(i_emb.get_shape().as_list()))
        # [b, h]
        print("j_emb 的维度是：{}".format(j_emb.get_shape().as_list()))


        #TODO ##############AFM######################
        #(None, 1)
        afm_emb = afm(sequence=h_emb, mode='att')

        ############AFM######################
        # -- fcn begin -------
        din_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')

        din_j = tf.concat([u_emb_j, j_emb, u_emb_j * j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)

        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
        x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]
        #d_layer_3_i(None,)，  i_b(None,),    tf.reshape(afm_emb, [-1]) (None,)


        # TODO ##########此处加了 tf.reshape(afm_emb, [-1])
        self.logits = i_b + d_layer_3_i + tf.reshape(afm_emb, [-1])
        #######################################################

        # self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.mf_auc = tf.reduce_mean(tf.cast(x > 0, dtype=tf.float64))
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
        self.score_i = tf.reshape(self.score_i, [-1, 1])
        self.score_j = tf.reshape(self.score_j, [-1, 1])
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
        # print self.p_and_n.get_shape().as_list()

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, l):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.y: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
            self.lr: l,
        })
        return loss

    def eval(self, sess, uij):
        u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })
        return u_auc, socre_p_and_n

    def test(self, sess, uij):
        return sess.run(self.logits_sub, feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def attention(queries, keys, keys_length):
    '''
      queries:     [B, H]
      keys:        [B, T, H]
      keys_length: [B]
    '''
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
    outputs = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]

    return outputs


# TODO 开始训练模型
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

train_batch_size = 32
test_batch_size = 512
predict_batch_size = 32
predict_users_num = 1000
predict_ads_num = 100

# with open("./small_din_datasets.pkl", 'rb') as f:
with open("../../datasets/dataset-100.pkl", 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)

best_auc = 0.0


def calc_auc(raw_arr):
    """Summary
    Args:
        raw_arr (TYPE): Description
    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d: d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None


def _auc_arr(score):
    score_p = score[:, 0]
    score_n = score[:, 1]
    # print "============== p ============="
    # print score_p
    # print "============== n ============="
    # print score_n
    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0, 1, s])
    for s in score_n.tolist():
        score_arr.append([1, 0, s])
    return score_arr


def _eval(sess, model):
    auc_sum = 0.0
    score_arr = []
    for _, uij in DataInputTest(test_set, test_batch_size):
        auc_, score_ = model.eval(sess, uij)
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])
    test_gauc = auc_sum / len(test_set)
    Auc = calc_auc(score_arr)
    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        # model.save(sess, 'save_path/ckpt')
    return test_gauc, Auc


def _test(sess, model):
    auc_sum = 0.0
    score_arr = []
    predicted_users_num = 0
    print("test sub items")
    for _, uij in DataInputTest(test_set, predict_batch_size):
        if predicted_users_num >= predict_users_num:
            break
        score_ = model.test(sess, uij)
        score_arr.append(score_)
        predicted_users_num += predict_batch_size
    return score_[0]


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = Model(user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model))
    sys.stdout.flush()
    lr = 1.0
    start_time = time.time()

    # TODO 一般跑完 10 个批次就差不多了
    for _ in range(10):

        random.shuffle(train_set)

        epoch_size = round(len(train_set) / train_batch_size)
        loss_sum = 0.0
        for _, uij in DataInput(train_set, train_batch_size):
            loss = model.train(sess, uij, lr)
            loss_sum += loss

            if model.global_step.eval() % 1000 == 0:
                test_gauc, Auc = _eval(sess, model)
                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                      (model.global_epoch_step.eval(), model.global_step.eval(),
                       loss_sum / 1000, test_gauc, Auc))
                sys.stdout.flush()
                loss_sum = 0.0

            if model.global_step.eval() % 336000 == 0:
                lr = 0.1

        print('Epoch %d DONE\tCost time: %.2f' %
              (model.global_epoch_step.eval(), time.time() - start_time))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()

    print('best test_gauc:', best_auc)
    sys.stdout.flush()

"""
    (1) 使用 din_datasets 数据集跑出来的最高指标大概在 87%

    (2) 使用 large_kindle_datasets 数据集，效果非常稳定，AUC 最高 91.16%，GAUC 最高 90.29%
"""
