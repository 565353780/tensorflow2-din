#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import pickle
import tensorflow as tf
from tqdm import tqdm

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#  tf.config.experimental.set_virtual_device_configuration(
#      gpus[0],
#      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])

from data import DataLoader, DataLoaderTest
from model import Base, DIN, DIEN
from utils import eval
from DatasetPklCreater import DatasetPklCreater

class DINTrainer:
    def __init__(self):
        self.lr = 0.1 # learning rate
        self.train_batch_size = 32 # batch size
        self.test_batch_size = 512 # batch size
        self.epochs = 100 # number of epochs
        self.print_step = 10000 # step size for print gauc log
        self.loss_print_step = min(1000, self.print_step) # step size for print loss log
        self.dataset_dir = "../datasets/raw_data/" # dataset path
        self.model_path = "./models/" # model load path
        self.log_path = "./logs/" # log path for tensorboard
        self.is_reuse = False
        self.multi_gpu = False
        self.user_count = 192403 # number of users
        self.item_count = 63001 # number of items
        self.cate_count = 801 # number of categories
        self.user_dim = 128 # dimension of user
        self.item_dim = 64 # dimension of item
        self.cate_dim = 64 # dimension of category
        self.dim_layers = [80, 40, 1]

        self.dataset_pkl_creater = DatasetPklCreater()

        self.pos_list_len_max = None
        self.use_din_source_method = None

        self.train_data = None
        self.test_data = None
        self.user_count = None
        self.item_count = None
        self.cate_count = None
        self.cate_list = None

        self.optimizer = None
        self.loss_metric = None
        self.auc_metric = None

        self.model = None

        self.method_list = [[
            "Source",
            "AFM-Add-to-Output",
            "AFM-Add-to-Attention-Output",
            "AFM-With-Candidate"
            ],[
                "Source",
                "Add-Conv2D-to-Attention"
            ]]
        self.method_idx = None
        self.method_name = None

        self.global_step = None
        self.best_loss= None
        self.best_auc = None
        self.last_global_step = None
        self.last_save_loss = None
        self.last_save_auc = None

        self.source_lr = None
        self.decay_rate = None
        self.decay_steps = None
        self.decayed_lr = None

        self.train_summary_writer = None
        return

    def print_tf_info(self):
        print(tf.__version__)
        print("GPU Available: ", tf.test.is_gpu_available())
        return True

    def create_dataset_pkl(self, pos_list_len_max, use_din_source_method):
        if not self.dataset_pkl_creater.load_remap_pkl():
            return False
        return self.dataset_pkl_creater.create_dataset_pkl(
            pos_list_len_max, use_din_source_method)

    def set_method(self, method_idx):
        self.method_idx = method_idx
        self.method_name = self.method_list[0][self.method_idx[0]] + "_" + \
            self.method_list[1][self.method_idx[1]]
        return True

    def get_model_name(self, step, loss, auc):
        save_model_name = "DIN"
        save_model_name += "_best_step_" + str(step)
        save_model_name += "_loss_" + str(float(loss))[:6]
        save_model_name += "_gauc_" + str(float(auc))[:6]
        save_model_name += ".ckpt"
        return save_model_name

    def load_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            train_set = pickle.load(f, encoding='latin1')
            test_set = pickle.load(f, encoding='latin1')
            cate_list = pickle.load(f, encoding='latin1')
            self.cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)
            self.user_count, self.item_count, self.cate_count = pickle.load(f)

        self.train_data = DataLoader(self.train_batch_size, train_set)
        self.test_data = DataLoaderTest(self.test_batch_size, test_set)
        return True

    def load_train_objects(self):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr, momentum=0.0)
        self.loss_metric = tf.keras.metrics.Sum()
        self.auc_metric = tf.keras.metrics.AUC()
        return True

    def set_learning_rate_param(self, source_lr, decay_rate, decay_steps):
        if source_lr is None:
            self.source_lr = self.lr
            self.decayed_lr = self.source_lr
        else:
            self.source_lr = source_lr
            self.decayed_lr = self.source_lr
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.decayed_lr, momentum=0.0)

        if decay_rate is None:
            self.decay_rate = 0.9
        else:
            self.decay_rate = decay_rate

        if decay_steps is None:
            self.decay_steps = self.train_data.epoch_size
        else:
            self.decay_steps = decay_steps

        return True

    def set_summary_writer(self):
        log_name = self.method_name
        log_name += "_PosListLenMax_" + str(self.pos_list_len_max)
        log_name += "_UseDinSourceMethod_" + str(self.use_din_source_method)
        log_name += "_Lr_" + str(self.source_lr)
        log_name += "_DecayRate_" + str(self.decay_rate)
        log_name += "_DecaySteps_" + str(self.decay_steps)

        self.train_summary_writer = tf.summary.create_file_writer(
            self.log_path + log_name)
        return True

    def load_model(self):
        #192403 63001 801 tf.Tensor([738 157 571 ...  63 674 351], shape=(63001,), dtype=int64)
        #print(user_count,item_count,cate_count,cate_list,"111")

        self.model = DIN(self.user_count, self.item_count, self.cate_count, self.cate_list,
                         self.user_dim, self.item_dim, self.cate_dim, self.dim_layers)

        self.model.set_method(self.method_idx)
        return True

    def load_trained_model_param(self):
        self.global_step = 0
        self.best_loss= 0.
        self.best_auc = 0.
        self.last_global_step = 0
        self.last_save_loss = 0.
        self.last_save_auc = 0.

        #  TODO: will always failed, return here
        return False

        last_save_model_name = None

        if os.path.exists(self.model_path + self.method_name + "/"):
            model_list = os.listdir(self.model_path + self.method_name + "/")
            for model_name in model_list:
                if ".index" in model_name:
                    model_name_split_list = model_name.split(".ckpt")[0].split("best_")[1].split("_")
                    current_auc = float(model_name_split_list[5])
                    if current_auc > self.last_save_auc:
                        self.last_global_step = int(model_name_split_list[1])
                        self.global_step = self.last_global_step + 1
                        self.last_save_loss = float(model_name_split_list[3])
                        self.best_loss = self.last_save_loss
                        self.last_save_auc = float(model_name_split_list[5])
                        self.best_auc = self.last_save_auc

            last_save_model_name = self.get_model_name(
                self.last_global_step, self.last_save_loss, self.last_save_auc)

            if self.last_save_auc > 0:
                try:
                    print("start load weights from :")
                    print(self.model_path + self.method_name + "/" + last_save_model_name)
                    self.model.load_weights(
                        self.model_path + self.method_name + "/" + last_save_model_name).expect_partial()
                    self.update_decay_lr()
                    return True
                except:
                    print("load weights failed, start trainning from step 0")
                    self.global_step = 0
                    self.best_loss = 0.
                    self.best_auc = 0.
                    self.last_global_step = 0
                    self.last_save_loss = 0.
                    self.last_save_auc = 0.
                    return False

        print("trained model not found, now will start trainning from step 0")
        return False

    def save_best_model(self):
        last_save_model_name = \
            self.get_model_name(self.last_global_step, self.last_save_loss, self.last_save_auc)
        if os.path.exists(self.model_path + self.method_name + "/"):
            saved_model_name_list = os.listdir(self.model_path + self.method_name + "/")
            for save_model_name in saved_model_name_list:
                if last_save_model_name in save_model_name:
                    os.remove(self.model_path + self.method_name + "/" + save_model_name)
        else:
            os.makedirs(self.model_path + self.method_name + "/")

        new_save_model_name = self.get_model_name(self.global_step, self.best_loss, self.best_auc)
        self.model.save_weights(self.model_path + self.method_name + "/" + new_save_model_name)
        self.last_global_step = self.global_step
        self.last_save_loss = self.best_loss
        self.last_save_auc = self.best_auc
        return True

    def update_decay_lr(self):
        min_lr = 0.0001
        if self.decayed_lr == min_lr:
            return True

        self.decayed_lr = self.source_lr * pow(self.decay_rate, (self.global_step / self.decay_steps))

        if self.decayed_lr < min_lr:
            self.decayed_lr = min_lr

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.decayed_lr, momentum=0.0)

        return True

    def init_env(self,
                 method_idx,
                 pos_list_len_max,
                 use_din_source_method,
                 source_lr=None,
                 decay_rate=None,
                 decay_steps=None):
        print("start create dataset...")
        self.pos_list_len_max = pos_list_len_max
        self.use_din_source_method = use_din_source_method
        if self.dataset_pkl_creater.is_dataset_exists(pos_list_len_max, use_din_source_method):
            print("model already exists, skip creating process")
        else:
            if not self.create_dataset_pkl(pos_list_len_max, use_din_source_method):
                return False
        print("SUCCESS!")

        print("start set trainning method...")
        if not self.set_method(method_idx):
            return False
        print("SUCCESS!")

        print("start load trainning and testing dataset...")
        if not self.load_dataset(
                "../datasets/" + self.dataset_pkl_creater.get_dataset_name(pos_list_len_max, use_din_source_method)):
            return False
        print("SUCCESS!")

        print("start load optimizer, loss_metric, auc_metric and lr...")
        if not self.load_train_objects():
            return False
        print("SUCCESS!")

        print("start set decayed lr param...")
        if not self.set_learning_rate_param(source_lr, decay_rate, decay_steps):
            return False
        print("SUCCESS!")

        print("start set summary writer...")
        if not self.set_summary_writer():
            return False
        print("SUCCESS!")

        print("start load model structure...")
        if not self.load_model():
            return False
        print("SUCCESS!")

        print("start load trained model param...")
        if not self.load_trained_model_param():
            return False
        print("SUCCESS!")

        return True

    # @tf.function
    def train_one_step(self, u, i, y, hist_i, sl):
        with tf.GradientTape() as tape:
            output,_ = self.model(u, i, hist_i, sl)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=output, labels=tf.cast(y, dtype=tf.float32)))
        gradient = tape.gradient(loss, self.model.trainable_variables)
        clip_gradient, _ = tf.clip_by_global_norm(gradient, 5.0)
        self.optimizer.apply_gradients(zip(clip_gradient, self.model.trainable_variables))

        self.loss_metric(loss)
        return True

    def train(self):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('lr', self.decayed_lr, step=self.global_step)

        for epoch in range(self.epochs):

            saved_loss_num = 0
            pbar = tqdm(total=self.print_step, desc="TRAIN")

            for step, (u, i, y, hist_i, sl) in enumerate(self.train_data, start=1):
                self.train_one_step(u, i, y, hist_i, sl)

                pbar.update(1)
                saved_loss_num += 1
                self.global_step += 1

                if self.global_step % self.loss_print_step == 0:
                    current_loss = self.loss_metric.result() / saved_loss_num
                    self.loss_metric.reset_states()
                    saved_loss_num = 0

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', current_loss, step=self.global_step)

                if self.global_step % self.print_step == 0:
                    pbar.close()

                    test_gauc, auc = eval(self.model, self.test_data)

                    print('Epoch %d Global_step %d\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                          (epoch, step, test_gauc, auc))

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('test_gauc', test_gauc, step=self.global_step)
                        tf.summary.scalar('lr', self.decayed_lr, step=self.global_step)

                    if self.best_auc < test_gauc:
                        self.best_loss = current_loss
                        self.best_auc = test_gauc

                        self.save_best_model()

                    pbar = tqdm(total=self.print_step, desc="TRAIN")

            self.loss_metric.reset_states()

            self.update_decay_lr()
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.decayed_lr, momentum=0.0)

            print('==== Epoch:', epoch, '-> Best test_gauc:', float(self.best_auc), "====")
        return True

if __name__ == '__main__':
    din_trainer = DINTrainer()
    din_trainer.init_env(method_idx=[0, 0],
                         pos_list_len_max=100,
                         use_din_source_method=True,
                         source_lr=0.1,
                         decay_rate=0.9,
                         decay_steps=None)
    din_trainer.train()

