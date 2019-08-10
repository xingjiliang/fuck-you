# coding=utf-8
import os
from absl import app
from absl import flags
from importlib import import_module
from src import config
from src import trainer
from src import tester

global model_config


def main(_):
    check_arguments()
    global model_config
    model_config = config.ModelConfiguration()
    model = import_module(os.path.join(config.MODEL_PATH, flags.FLAGS.model)).Model(model_config)
    trainer.train(model, flags.FLAGS.train_data_path)
    tester.test(model, flags.FLAGS.test_data_path)


def check_arguments():
    FLAGS = flags.FLAGS
    if os.path.exists(os.path.join(config.MODEL_PARAMETER_PATH, FLAGS.application_name)):
        print("模型参数文件夹{}已存在".format(FLAGS.application_name))
        return False


def _init():
    _define_flags()


def _define_flags():
    flags.DEFINE_string(name="application_name", short_name="a", default="DOGGY", help="模型参数代号")
    flags.DEFINE_string(name="model", short_name="m", default="base_model", help="选择models目录下的模型")
    flags.DEFINE_string(name="train_data", short_name="n", default=None, help="训练集文件名-直接输入data下的文件名即可")
    flags.DEFINE_string(name="test_data", short_name="t", default=None, help="测试集文件名")
    flags.DEFINE_integer(name="epoch_num", short_name="e", default=None, help="训练集迭代次数")
    flags.DEFINE_integer(name="batch_size", short_name="b", default=None, help="batch_size")
    flags.DEFINE_float(name="learn_rate", short_name="l", default=None, help="学习率")
    flags.DEFINE_string(name="config", short_name="c", default="youtube_deep_recall_model.cfg", help="模型配置文件")


if __name__ == "__main__":
    _init()
    app.run(main)
