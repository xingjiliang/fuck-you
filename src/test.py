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

import config
import data_utils

LOG = config.logging.getLogger('test')


def test(global_config):

    test_dataset = data_utils.from_text_line_file(global_config, False)
    sample = test_dataset.make_one_shot_iterator().get_next()

    with tf.variable_scope(global_config.model_name, reuse=tf.AUTO_REUSE):

        model = import_module("models." + global_config.model_name).Model(global_config, sample, np.load(
            config.INFO_INPUT_EMBEDDINGS_PATH), False)
        saver = tf.train.Saver(max_to_keep=None)
        model_parameters_path = "{}".format(os.path.join(config.MODEL_PARAMETERS_PATH, global_config.application_name))
        auc_op_ts, auc_value_ts = tf.metrics.auc(model.labels, tf.sigmoid(model.logits))

        # batch_accuracy_num_node = tf.reduce_sum(tf.cast(tf.equal(model.labels, tf.arg_max(model.logits, 1)), tf.int32))
        fetches = [auc_value_ts, model.loss, model.labels, tf.sigmoid(model.logits)]
        global_labels = []
        global_logits = []
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, model_parameters_path)
            try:
                while True:
                    sess.run(auc_op_ts)
                    auc, loss, labels, logits = sess.run(fetches)
                    global_labels.append(labels)
                    global_logits.append(logits)
                    LOG.info('损失={},AUC={}'.format(loss, auc))
            except tf.errors.OutOfRangeError:
                global_auc_op_ts, global_auc_value_ts = tf.metrics.auc(np.concatenate(global_labels),
                                                                       np.concatenate(global_logits))
                sess.run(tf.local_variables_initializer())
                sess.run(auc_op_ts)
                global_auc = sess.run(global_auc_value_ts)
                LOG.info('all AUC={}'.format(global_auc))
            sess.close()


def check_arguments(FLAGS):
    if not os.path.exists(os.path.join(config.MODEL_PARAMETERS_PATH, FLAGS.application_name + '.meta')):
        print("{}模型不存在".format(FLAGS.application_name))
        return False
    if not FLAGS.test_dataset_path:
        print("请输入测试集文件名")
        return False
    if not os.path.exists(os.path.join(config.DATA_PATH, FLAGS.test_dataset_path)):
        print("测试集{}不存在".format(FLAGS.test_dataset_path))
        return False
    return True


def main(_):
    FLAGS = flags.FLAGS
    if not check_arguments(FLAGS):
        exit(1)
    global_config = config.Configuration(FLAGS)
    test(global_config)


def _define_flags():
    flags.DEFINE_string(name="application_name", short_name="a", default="DOGGY", help="模型参数代号")
    flags.DEFINE_string(name="model_name", short_name="m", default="base", help="选择models目录下的模型")
    flags.DEFINE_string(name="train_dataset_path", short_name="n", default='none', help="训练集文件名")
    flags.DEFINE_string(name="test_dataset_path", short_name="t", default=None, help="测试集文件名")
    flags.DEFINE_integer(name="epoch_num", short_name="e", default=None, help="训练集迭代次数")
    flags.DEFINE_integer(name="batch_size", short_name="b", default=65536, help="batch_size")
    flags.DEFINE_float(name="learn_rate", short_name="l", default=None, help="学习率")
    flags.DEFINE_string(name="model_config_file", short_name="c", default="im_click_rate_prediction_model.cfg", help="模型配置文件")
    flags.DEFINE_string(name="feature_config_file", short_name="f", default="im_features_unpreprocessed_continuous.json", help="特征配置文件")


if __name__ == "__main__":
    _define_flags()
    app.run(main)
