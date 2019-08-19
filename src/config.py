# coding=utf-8
import configparser
import os
import json

from absl import flags
from absl import app

PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
SOURCE_CODE_PATH = os.path.join(PROJECT_ROOT_PATH, "src")
RESOURCES_PATH = os.path.join(PROJECT_ROOT_PATH, "resources")
DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data")
MODEL_PARAMETERS_PATH = os.path.join(PROJECT_ROOT_PATH, "model_parameters")
MODELS_PATH = os.path.join(SOURCE_CODE_PATH, "models")
INFO_INPUT_EMBEDDINGS_TEXT_FILE_PATH = os.path.join(DATA_PATH, "input_vector_sample")
INFO_INPUT_EMBEDDINGS_PATH = os.path.join(DATA_PATH, "info_input_embeddings.npy")
DEFAULT_CONFIG_FILE_PATH = os.path.join(RESOURCES_PATH, "youtube_deep_recall_model.cfg")

INPUT_FEATURE_SPACE = "input_space"
INPUT_FEATURE_INDEX = 'index'
INPUT_FEATURE_FORM = 'form'
INPUT_FEATURE_TYPE = 'feature_type'
FEATURE_SPACE = 'space'
INPUT_FEATURE_DIM = 'input_dim'
EMBEDDING_DIM = 'embedding_dim'
INITIALIZER = 'initializer'

class Configuration:
    def __init__(self, FLAGS):
        self.application_name = FLAGS.application_name
        self.model_name = FLAGS.model_name
        self.train_dataset_path = os.path.join(DATA_PATH, FLAGS.train_dataset_path)
        self.test_dataset_path = os.path.join(DATA_PATH, FLAGS.test_dataset_path)
        self.feature_index_type_map = {}
        self.attribute_raw_dim_map = {}
        # 对指定特征设置维度
        self.attribute_embedding_dim_map = {}
        self.from_model_config_file(os.path.join(RESOURCES_PATH, FLAGS.model_config_file))
        self.from_feature_config_file(os.path.join(RESOURCES_PATH, FLAGS.feature_config_file))
        self.from_command_line_arguments(FLAGS)

    def from_model_config_file(self, file_path):
        conf = configparser.ConfigParser()
        conf.read(file_path)
        for option in conf.options("model_hyper_parameters"):
            exec("self." + option + " = " + conf.get("model_hyper_parameters", option))

    def from_feature_config_file(self, feature_config_file):
        self.feature_config = json.loads(open(os.path.join(RESOURCES_PATH, feature_config_file)).read())

    def from_command_line_arguments(self, FLAGS):
        for option in ["epoch_num", "batch_size", "learn_rate"]:
            value = eval("FLAGS." + option)
            if value:
                exec("self." + option + " = " + str(value))

    def __str__(self):
        string = ""
        for option, argument in self.__dict__.items():
            string += ("--{} {}\n".format(option, argument))
        return string


def main(_):
    m = Configuration(flags.FLAGS)
    print(m)


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
