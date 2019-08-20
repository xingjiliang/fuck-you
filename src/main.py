# coding=utf-8
import os
from absl import app
from absl import flags
from importlib import import_module
import config
import train
import test


def main(_):
    FLAGS = flags.FLAGS
    if not check_arguments(FLAGS):
        exit(1)
    global_config = config.Configuration(FLAGS)
    # print(global_config)
    train.train(global_config)
    test.test(global_config)


def check_arguments(FLAGS):
    if os.path.exists(os.path.join(config.MODEL_PARAMETERS_PATH, FLAGS.application_name)):
        print("模型参数文件夹{}已存在".format(FLAGS.application_name))
        return False
    if not FLAGS.train_dataset_path or not FLAGS.train_dataset_path:
        print("请输入训练集与测试集文件名")
        return False
    if not os.path.exists(os.path.join(config.DATA_PATH, FLAGS.train_dataset_path)):
        print("训练集{}不存在".format(FLAGS.train_dataset_path))
        return False
    if not os.path.exists(os.path.join(config.DATA_PATH, FLAGS.test_dataset_path)):
        print("测试集{}不存在".format(FLAGS.test_dataset_path))
        return False
    return True


def _define_flags():
    flags.DEFINE_string(name="application_name", short_name="a", default="DOGGY", help="模型参数代号")
    flags.DEFINE_string(name="model_name", short_name="m", default="base", help="选择models目录下的模型")
    flags.DEFINE_string(name="train_dataset_path", short_name="n", default=None, help="训练集文件名")
    flags.DEFINE_string(name="test_dataset_path", short_name="t", default=None, help="测试集文件名")
    flags.DEFINE_integer(name="epoch_num", short_name="e", default=None, help="训练集迭代次数")
    flags.DEFINE_integer(name="batch_size", short_name="b", default=None, help="batch_size")
    flags.DEFINE_float(name="learn_rate", short_name="l", default=None, help="学习率")
    flags.DEFINE_string(name="model_config_file", short_name="c", default="youtube_deep_recall_model.cfg", help="模型配置文件")
    flags.DEFINE_string(name="feature_config_file", short_name="f", default="features.json", help="特征配置文件")


if __name__ == "__main__":
    _define_flags()
    app.run(main)
