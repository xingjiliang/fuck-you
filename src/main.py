# coding=utf-8
import os
from absl import app
from absl import flags
from importlib import import_module
from src import config
from src import train
from src import test


def main(_):
    check_arguments()
    model_config = config.ModelConfiguration()
    sess = train.train(flags.model, model_config, flags.application_name, flags.train_dataset_path)
    test.test(flags.model, model_config, flags.application_name, flags.test_dataset_path, sess)


def check_arguments():
    FLAGS = flags.FLAGS
    if os.path.exists(os.path.join(config.MODEL_PARAMETER_PATH, FLAGS.application_name)):
        print("模型参数文件夹{}已存在".format(FLAGS.application_name))
        return False


def _define_flags():
    flags.DEFINE_string(name="application_name", short_name="a", default="DOGGY", help="模型参数代号")
    flags.DEFINE_string(name="model", short_name="m", default="base_model", help="选择models目录下的模型")
    flags.DEFINE_string(name="train_dataset_path", short_name="n", default=None, help="训练集文件名")
    flags.DEFINE_string(name="test_dataset_path", short_name="t", default=None, help="测试集文件名")
    flags.DEFINE_integer(name="epoch_num", short_name="e", default=None, help="训练集迭代次数")
    flags.DEFINE_integer(name="batch_size", short_name="b", default=None, help="batch_size")
    flags.DEFINE_float(name="learn_rate", short_name="l", default=None, help="学习率")
    flags.DEFINE_string(name="config", short_name="c", default="youtube_deep_recall_model.cfg", help="模型配置文件")


if __name__ == "__main__":
    _define_flags()
    app.run(main)
