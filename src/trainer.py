# coding=utf-8
"""
生成样本
训练模型
保存模型参数
通过使用make_initializable_iterator方便地切换数据集……
"""
import tensorflow as tf
from src import data_preprocessor


def train(model, train_data_path):
    text_line_dataset = tf.data.TextLineDataset(train_data_path)
    for feature in model.model_config.feature_index_type_map:
        index, feature_nature, feature_type, attribute = model.model_config.feature_index_type_map[feature]

    padded_shapes = ()
    iterator = text_line_dataset.map(data_preprocessor.to_instance).padded_batch(64, ([], [], [], [], [], [], [], [50], [50], [50], [50], [], [], [])).repeat(5).shuffle(10000)\
        .make_initializable_iterator()
    train_batch = iterator.get_next()
    sess = tf.Session()
    while True:
        sess.run()
