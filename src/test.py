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


def test(global_config):

    test_dataset = data_utils.from_text_line_file(global_config, False)
    sample = test_dataset.make_one_shot_iterator().get_next()

    with tf.variable_scope(global_config.model_name, reuse=tf.AUTO_REUSE):

        model = import_module("models." + global_config.model_name).Model(global_config, sample, np.load(
            config.INFO_INPUT_EMBEDDINGS_PATH), False)
        saver = tf.train.Saver(max_to_keep=None)
        save_path = "{}".format(os.path.join(config.MODEL_PARAMETERS_PATH, global_config.application_name))
        auc_value, auc_op = tf.metrics.auc(model.labels, model.predictions)

        # batch_accuracy_num_node = tf.reduce_sum(tf.cast(tf.equal(model.labels, tf.arg_max(model.logits, 1)), tf.int32))
        fetches = [auc_value, model.loss]
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, save_path)
            try:
                while True:
                    sess.run(auc_op)
                    batch_auc, loss = sess.run(fetches)
                    print(batch_auc)
            except tf.errors.OutOfRangeError:
                print(batch_auc)
