# coding=utf-8
"""
生成样本
训练模型
保存模型参数
"""
import tensorflow as tf
from src import data_preprocessor


def train(model, train_data_path):
    text_line_dataset = tf.data.TextLineDataset(train_data_path)
    iterator = text_line_dataset.map(data_preprocessor.to_instance).repeat(5).batch(64).shuffle(10000)\
        .make_initializable_iterator()
    train_batch = iterator.get_next()
    sess = tf.Session()
    while True:
        sess.run()

