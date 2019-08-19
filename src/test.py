# coding=utf-8
"""
通过data_utils得到样本
训练模型
保存模型参数
通过使用make_initializable_iterator方便地切换数据集……
"""
import os
from importlib import import_module

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
import data_utils
import config


def test(global_config, sess):

    test_dataset = data_utils.from_text_line_file(global_config, False)
    sample = test_dataset.make_one_shot_iterator().get_next()

    with tf.variable_scope(global_config.model_name, reuse=tf.AUTO_REUSE):
        model = import_module(os.path.join(config.MODELS_PATH, global_config.model_name)).Model(global_config, sample, np.load(
            config.INFO_INPUT_EMBEDDINGS_PATH), False)
        accuracy_num = 0
        batch_accuracy_num_node = tf.reduce_sum(tf.cast(tf.equal(model.labels, tf.arg_max(model.logits, 1)), tf.int32))
        fetches = [batch_accuracy_num_node, model.loss]
        try:
            while True:
                batch_accuracy_num, loss = sess.run(fetches)
                accuracy_num += batch_accuracy_num[0]
        except tf.errors.OutOfRangeError:
            print(accuracy_num)
