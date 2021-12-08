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
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])

from data import DataLoader, DataLoaderTest
from model import Base, DIN, DIEN
from utils import eval

class DINTrainer:
    def __init__(self):
        self.lr=0.1
        self.train_batch_size=32
        self.test_batch_size=512
        self.epochs=10000
        self.print_step=10000
        self.model_path="./models/"
        self.log_path="./logs/"
        self.is_reuse=False
        self.multi_gpu=False
        self.user_count=192403
        self.item_count=63001
        self.cate_count=801
        self.user_dim=128
        self.item_dim=64
        self.cate_dim=64
        self.dim_layers=[80, 40, 1]

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

        self.method_list = [
            "Source",
            "AFM-Add-to-Output",
            "AFM-Add-to-Attention-Output",
            "AFM-With-Candidate"]
        self.method_idx = None
        self.method_name = None

        self.global_step = None
        self.best_loss= None
        self.best_auc = None
        self.last_global_step = None
        self.last_save_loss = None
        self.last_save_auc = None

        self.train_summary_writer = None

        self.source_lr = None
        self.decay_rate = None
        self.decay_steps = None
        self.decayed_lr = None
        return

    def print_tf_info(self):
        print(tf.__version__)
        print("GPU Available: ", tf.test.is_gpu_available())
        return True

    def print_method_list(self):
        print(self.method_list)
        return True

    def set_method(self, method_idx):
        self.method_idx = method_idx
        self.method_name = self.method_list[self.method_idx]

        self.train_summary_writer = tf.summary.create_file_writer(
            self.log_path + self.method_name)
        return True

    def get_model_name(self, step, loss, auc):
        save_model_name = "DIN_best_step_" + str(step) + \
            "_loss_" + str(float(loss))[:6] + \
            "_gauc_" + str(float(auc))[:6] + ".ckpt"
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
        self.source_lr = self.lr
        self.decayed_lr = self.source_lr
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.decayed_lr, momentum=0.0)
        self.loss_metric = tf.keras.metrics.Sum()
        self.auc_metric = tf.keras.metrics.AUC()
        return True

    def set_decayed_lr_param(self, decay_rate, decay_steps):
        if decay_rate is None:
            self.decay_rate = 0.999
        else:
            self.decay_rate = decay_rate

        if decay_steps is None:
            self.decay_steps = self.train_data.epoch_size
        else:
            self.decay_steps = decay_steps

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

        if os.path.exists(self.model_path + self.method_name + "/"):
            model_list = os.listdir(self.model_path + self.method_name + "/")
            for model_name in model_list:
                if "DIN" == model_name[:3]:
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
                        self.model_path + self.method_name + "/" + last_save_model_name)
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
        return True

    def save_best_model(self):
        last_save_model_name = \
            self.get_model_name(self.last_global_step, self.last_save_loss, self.last_save_auc)
        if os.path.exists(self.model_path + self.method_name + "/"):
            saved_model_name_list = os.listdir(self.model_path + self.method_name + "/")
            for save_model_name in saved_model_name_list:
                if last_save_model_name in save_model_name:
                    os.remove(self.model_path + self.method_name + "/" + save_model_name)

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
            return True

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.decayed_lr, momentum=0.0)

        print("trainning with decayed_lr =", self.decayed_lr)
        return True

    def init_env(self, method_idx, dataset_path, decay_rate=None, decay_steps=None):
        print("start set trainning method...")
        if not self.set_method(method_idx):
            return False
        print("SUCCESS!")

        print("start load trainning and testing dataset...")
        if not self.load_dataset(dataset_path):
            return False
        print("SUCCESS!")

        print("start load optimizer, loss_metric, auc_metric and lr...")
        if not self.load_train_objects():
            return False
        print("SUCCESS!")

        print("start set decayed lr param...")
        if not self.set_decayed_lr_param(decay_rate, decay_steps):
            return False
        print("SUCCESS!")

        print("start load model structure...")
        if not self.load_model():
            return False
        print("SUCCESS!")

        print("start load trained model weights...")
        if not self.load_trained_model_param():
            print("NOTE: this might be a tf2 keras official bug")
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
        for epoch in range(self.epochs):

            pbar = tqdm(total=self.print_step, desc="TRAIN")

            for step, (u, i, y, hist_i, sl) in enumerate(self.train_data, start=1):
                if not self.train_one_step(u, i, y, hist_i, sl):
                    return False

                pbar.update(1)

                if step % self.print_step == 0:
                    pbar.close()

                    test_gauc, auc = eval(self.model, self.test_data)
                    current_loss = self.loss_metric.result() / self.print_step

                    print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                          (epoch, step, current_loss, test_gauc, auc))

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', current_loss, step=self.global_step)
                        tf.summary.scalar('test_gauc', test_gauc, step=self.global_step)
                        self.global_step += self.print_step

                    if self.best_auc < test_gauc:
                        self.best_loss = current_loss
                        self.best_auc = test_gauc

                    self.save_best_model()

                    self.loss_metric.reset_states()

                    pbar = tqdm(total=self.print_step, desc="TRAIN")

            self.loss_metric.reset_states()

            self.update_decay_lr()

            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.decayed_lr, momentum=0.0)

            print('==== Epoch:', epoch, '-> Best test_gauc:', self.best_auc, "====")
        return True

if __name__ == '__main__':
    din_trainer = DINTrainer()
    din_trainer.init_env(method_idx=3, dataset_path="../datasets/dataset-100.pkl")
    din_trainer.train()

