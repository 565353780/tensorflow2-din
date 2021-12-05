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

#  from DIEN.config import argparser
#  from DIEN.data import get_dataloader
#  from DIEN.model import Base, DIN, DIEN
#  from DIEN.utils import eval

from config import argparser
from data import get_dataloader
from model import Base, DIN, DIEN
from utils import eval

# Config
# print(tf.__version__)
# print("GPU Available: ", tf.test.is_gpu_available())

args = argparser()

# Data Load
train_data, test_data, \
user_count, item_count, cate_count, \
cate_list = get_dataloader(args.train_batch_size, args.test_batch_size)

# Loss, Optim
optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.0)
loss_metric = tf.keras.metrics.Sum()
auc_metric = tf.keras.metrics.AUC()


#192403 63001 801 tf.Tensor([738 157 571 ...  63 674 351], shape=(63001,), dtype=int64)
#print(user_count,item_count,cate_count,cate_list,"111")
# Model
model = DIN(user_count, item_count, cate_count, cate_list,
             args.user_dim, args.item_dim, args.cate_dim, args.dim_layers)


# @tf.function
def train_one_step(u,i,y,hist_i,sl):
    with tf.GradientTape() as tape:
        output,_ = model(u,i,hist_i,sl)
        loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=output,
                                                        labels=tf.cast(y, dtype=tf.float32)))
    gradient = tape.gradient(loss, model.trainable_variables)
    clip_gradient, _ = tf.clip_by_global_norm(gradient, 5.0)
    optimizer.apply_gradients(zip(clip_gradient, model.trainable_variables))

    loss_metric(loss)

# Train
def train(optimizer):
    method_list = ["Source", "AFM-Add-to-Output", "AFM-Add-to-Attention-Output", "AFM-With-Candidate"]
    method_name = method_list[3]

    method_train_episode = 0
    folder_list = os.listdir("./logs/")
    for folder_name in folder_list:
        folder_name_split_list = folder_name.split("_")
        if len(folder_name_split_list) < 2:
            continue
        if folder_name_split_list[0] == method_name:
            method_exist_train_episode = int(folder_name_split_list[1])
            if method_exist_train_episode > method_train_episode:
                method_train_episode = method_exist_train_episode
    method_train_episode += 1

    global_step = 0

    # Board
    train_summary_writer = tf.summary.create_file_writer(args.log_path + method_name + "_" + str(method_train_episode))

    best_loss= 0.
    best_auc = 0.
    start_time = time.time()
    for epoch in range(args.epochs):
        for step, (u, i, y, hist_i, sl) in enumerate(train_data, start=1):
            train_one_step(u, i, y, hist_i, sl)

            if step % args.print_step == 0:
                test_gauc, auc = eval(model, test_data)
                current_loss = loss_metric.result() / args.print_step

                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                      (epoch, step, current_loss, test_gauc, auc))

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', current_loss, step=global_step)
                    tf.summary.scalar('test_gauc', test_gauc, step=global_step)
                    global_step += 1

                if best_auc < test_gauc:
                    best_loss = current_loss
                    best_auc = test_gauc
                    model.save_weights(args.model_path + method_name +
                        '_best_EPOCH_' + str(epoch) + '_STEP_' + str(step) + '.ckpt')
                loss_metric.reset_states()

        loss_metric.reset_states()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

        print('Epoch %d DONE\tCost time: %.2f' % (epoch, time.time()-start_time))
    print('Best test_gauc: ', best_auc)


# Main
if __name__ == '__main__':
    train(optimizer)
