import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Created on August 3, 2020
Updated on May 19, 2021

train AFM model

@author: Ziyao Geng(zggzy1996@163.com)
"""

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from AFM.model import AFM
from data_process.criteo import create_criteo_dataset
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    # If you have GPU, and the value is GPU serial number.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    file = '../datasets/Criteo/criteo_sampled_data.csv'
    read_part = False
    sample_num = 5000000
    test_size = 0.2

    embed_dim = 32
    att_vector = 8
    mode = 'att'  # 'max', 'avg'
    dropout = 0.5
    activation = 'relu'
    embed_reg = 1e-5

    learning_rate = 0.01
    batch_size = 4096
    epochs = 20

    # ========================== Create dataset =======================
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         embed_dim=embed_dim,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    #(480000, 39),(480000,)
    test_X, test_y = test
    #(120000, 39),(120000,)
    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = AFM(feature_columns, mode, att_vector, activation, dropout, embed_reg)
        # model.summary()
        # =========================Compile============================
        model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                      metrics=[AUC()])
    # ============================model checkpoint======================
    # check_path = 'save/afm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ===========================Fit==============================

    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)], # checkpoint,
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])