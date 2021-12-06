import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
'''0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed'''
import time
import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])

from config import argparser
from data import get_dataloader
from model import Base, DIN, DIEN
from utils import eval

from tqdm import tqdm

class DINTrainer:
    def __init__(self):
        self.args = argparser()

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
        self.method_name = None

        self.global_step = None
        self.best_loss= None
        self.best_auc = None
        self.last_global_step = None
        self.last_save_loss = None
        self.last_save_auc = None

        self.train_summary_writer = None
        return

    def print_tf_info(self):
        print(tf.__version__)
        print("GPU Available: ", tf.test.is_gpu_available())
        return True

    def set_method(self, method_idx):
        self.method_name = self.method_list[method_idx]

        self.train_summary_writer = tf.summary.create_file_writer(
            self.args.log_path + self.method_name)
        return True

    def get_model_name(self, step, loss, auc):
        save_model_name = "DIN_best_step_" + str(step) + \
            "_loss_" + str(float(loss))[:6] + \
            "_gauc_" + str(float(auc))[:6] + ".ckpt"
        return save_model_name

    def load_dataset(self):
        self.train_data, self.test_data, self.user_count, self.item_count, self.cate_count, self.cate_list = \
            get_dataloader(self.args.train_batch_size, self.args.test_batch_size)
        return True

    def load_train_objects(self):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.args.lr, momentum=0.0)
        self.loss_metric = tf.keras.metrics.Sum()
        self.auc_metric = tf.keras.metrics.AUC()
        return True

    def load_model(self):
        #192403 63001 801 tf.Tensor([738 157 571 ...  63 674 351], shape=(63001,), dtype=int64)
        #print(user_count,item_count,cate_count,cate_list,"111")
        self.model = DIN(self.user_count, self.item_count, self.cate_count, self.cate_list,
                    self.args.user_dim, self.args.item_dim, self.args.cate_dim, self.args.dim_layers)
        return True

    def load_trained_model_param(self):
        self.global_step = 0
        self.best_loss= 0.
        self.best_auc = 0.
        self.last_global_step = 0
        self.last_save_loss = 0.
        self.last_save_auc = 0.

        if os.path.exists(self.args.model_path + self.method_name + "/"):
            model_list = os.listdir(self.args.model_path + self.method_name + "/")
            for model_name in model_list:
                if "DIN" == model_name[:3]:
                    model_name_split_list = model_name.split(".ckpt")[0].split("best_")[1].split("_")
                    self.last_global_step = int(model_name_split_list[1])
                    self.global_step = self.last_global_step + 1
                    self.last_save_loss = float(model_name_split_list[3])
                    self.best_loss = self.last_save_loss
                    self.last_save_auc = float(model_name_split_list[5])
                    self.best_auc = self.last_save_auc

                    last_save_model_name = self.get_model_name(
                        self.last_global_step, self.last_save_loss, self.last_save_auc)

                    try:
                        self.model.load_weights(
                            self.args.model_path + self.method_name + "/" + last_save_model_name)
                        print("load trained model success")
                        return True
                    except:
                        print("load trained model failed, will start from step 0")
                        print("this might be a tf2 keras' official bug")
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
        if os.path.exists(self.args.model_path + self.method_name + "/"):
            saved_model_name_list = os.listdir(self.args.model_path + self.method_name + "/")
            for save_model_name in saved_model_name_list:
                if last_save_model_name in save_model_name:
                    os.remove(self.args.model_path + self.method_name + "/" + save_model_name)

        new_save_model_name = self.get_model_name(self.global_step, self.best_loss, self.best_auc)
        self.model.save_weights(self.args.model_path + self.method_name + "/" + new_save_model_name)
        self.last_global_step = self.global_step
        self.last_save_loss = self.best_loss
        self.last_save_auc = self.best_auc
        return True

    def init_env(self, method_idx):
        if not self.set_method(method_idx):
            return False

        if not self.load_dataset():
            return False

        if not self.load_train_objects():
            return False

        if not self.load_model():
            return False

        if not self.load_trained_model_param():
            return False

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
        start_time = time.time()
        for epoch in range(self.args.epochs):

            pbar = tqdm(total=self.args.print_step, desc="TRAIN")

            for step, (u, i, y, hist_i, sl) in enumerate(self.train_data, start=1):
                if not self.train_one_step(u, i, y, hist_i, sl):
                    return False

                pbar.update(1)

                if step % self.args.print_step == 0:
                    pbar.close()

                    test_gauc, auc = eval(self.model, self.test_data)
                    current_loss = self.loss_metric.result() / self.args.print_step

                    print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                          (epoch, step, current_loss, test_gauc, auc))

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', current_loss, step=self.global_step)
                        tf.summary.scalar('test_gauc', test_gauc, step=self.global_step)
                        self.global_step += 1

                    if self.best_auc < test_gauc:
                        self.best_loss = current_loss
                        self.best_auc = test_gauc

                    self.save_best_model()

                    self.loss_metric.reset_states()

                    pbar = tqdm(total=self.args.print_step, desc="TRAIN")

            self.loss_metric.reset_states()
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

            print('Epoch %d DONE\tCost time: %.2f' % (epoch, time.time()-start_time))
        print('Best test_gauc: ', self.best_auc)
        return True

if __name__ == '__main__':
    din_trainer = DINTrainer()
    din_trainer.init_env(method_idx=3)
    din_trainer.train()

