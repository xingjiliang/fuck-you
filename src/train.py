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


FLAGS = flags.FLAGS


def train(global_config):
    train_dataset = data_utils.from_text_line_file(global_config, True)
    sample = train_dataset.make_one_shot_iterator().get_next()

    with tf.variable_scope(global_config.model_name, reuse=False):
        model = import_module("models." + global_config.model_name).Model(global_config, sample, np.load(
            config.INFO_INPUT_EMBEDDINGS_PATH), True)
        saver = tf.train.Saver(max_to_keep=None)
        save_path = os.path.join(config.MODEL_PARAMETERS_PATH, global_config.application_name)
        global_step = tf.train.get_or_create_global_step()
        op = tf.train.AdamOptimizer(global_config.learn_rate).minimize(model.loss, global_step=global_step)

        auc_op_ts, auc_value_ts = tf.metrics.auc(model.labels, tf.sigmoid(model.logits))

        fetches = [model.loss, auc_value_ts, model.labels, tf.sigmoid(model.logits), op]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            try:
                while True:
                    sess.run(auc_op_ts)
                    loss, auc, labels, preds, op = sess.run(fetches)
                    print("损失={},AUC={}".format(loss, auc))
                    print("标签={}".format(np.concatenate([labels.reshape([-1, 1]), preds.reshape([-1, 1])], 1)))
            except tf.errors.OutOfRangeError:
                saver.save(sess, save_path)
                # saver.save(sess, save_path, global_step=global_step)

# def main(_):
#     pass
#
#
# def _define_flags():
#     flags.DEFINE_string(name="application_name", short_name="a", default="DOGGY", help="模型参数代号")
#     flags.DEFINE_string(name="model", short_name="m", default="base_model", help="选择models目录下的模型")
#     flags.DEFINE_string(name="train_data", short_name="n", default=None, help="训练集文件名-直接输入data下的文件名即可")
#     flags.DEFINE_string(name="test_data", short_name="t", default=None, help="测试集文件名")
#     flags.DEFINE_integer(name="epoch_num", short_name="e", default=None, help="训练集迭代次数")
#     flags.DEFINE_integer(name="batch_size", short_name="b", default=None, help="batch_size")
#     flags.DEFINE_float(name="learn_rate", short_name="l", default=None, help="学习率")
#     flags.DEFINE_string(name="config", short_name="c", default="youtube_deep_recall_model.cfg", help="模型配置文件")
#
#
# if __name__ == "__main__":
#     _define_flags()
#     app.run(main)
