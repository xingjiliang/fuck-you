# coding=utf-8
import os
import configparser
from absl import app
from absl import flags

conf = configparser.ConfigParser()
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
SOURCE_CODE_PATH = os.path.join(PROJECT_ROOT_PATH, "src")
RESOURCES_PATH = os.path.join(PROJECT_ROOT_PATH, "resources")
DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data")
MODEL_PARAMETER_PATH = os.path.join(PROJECT_ROOT_PATH, "model_parameters")
DEFAULT_MODEL_CONFIG_PATH = os.path.join(RESOURCES_PATH, "default_model_config.ini")
MODEL_PATH = os.path.join(SOURCE_CODE_PATH, "models")
INFO_INPUT_EMBEDDINGS_TEXT_FILE_PATH = os.path.join(DATA_PATH, "input_vector_sample")
INFO_INPUT_EMBEDDINGS_PATH = os.path.join(DATA_PATH, "info_input_embeddings.npy")
SINGLE_FEATURE_LIST = ["slot", "os", "selecttag", "education", "gender", "workedyears", "age_split"]
UNORDERED_FEATURE_LIST = ["targetcateid", "userportraittag"]
ORDERED_FEATURE_LIST = ["click_history_list", "delivery_history_list"]
TARGET_FEATURE_LIST = ["infoid", "clicktag", "dlytag"]
FEATURE_LIST = SINGLE_FEATURE_LIST + UNORDERED_FEATURE_LIST + ORDERED_FEATURE_LIST + TARGET_FEATURE_LIST


class ModelConfiguration:
    """
    后期添加上特征配置
    """
    def __init__(self):
        self.epoch_num = 3
        self.batch_size = 64
        self.learn_rate = 1.0
        self.info_size = 99
        self.info_embedding_size = 50
        self.default_embedding_size = 5
        self.max_sequence_size = 50
        # 这里可选将所有属性键值都加入，然后下面初始化时判断是否包含该键
        self.attribute_embedding_size_map = {"default": 10}

    def from_config_file(self, file_path):
        conf.read(file_path)
        for option in conf.options("model_configuration"):
            try:
                if option in self.__dict__:
                    exec("self." + option + " = " + conf.get("model_configuration", option))
                elif "attribute_embedding_size" == (option.split('.')[0]):
                    self.attribute_embedding_size_map[option.split('.')[1]] = conf.get("model_configuration", option)
                else:
                    raise AttributeError
            except AttributeError:
                exit(1)

    def from_command_line_arguments(self, app):
        """
        :param app: absl.app
        :return:
        """
        pass

    def __str__(self):
        string = ""
        for option, argument in self.__dict__.items():
            string += ("--{} {}\n".format(option, argument))
        return string


if __name__ == "__main__":
    m = ModelConfiguration()
    m.from_config_file(DEFAULT_MODEL_CONFIG_PATH)
    print(m)

