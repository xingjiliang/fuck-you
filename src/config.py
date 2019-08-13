# coding=utf-8
import configparser
import os

from absl import flags

project_root_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
source_code_path = os.path.join(project_root_path, "src")
resources_path = os.path.join(project_root_path, "resources")
data_path = os.path.join(project_root_path, "data")
MODEL_PARAMETERS_PATH = os.path.join(project_root_path, "model_parameters")
MODELS_PATH = os.path.join(source_code_path, "models")
INFO_INPUT_EMBEDDINGS_TEXT_FILE_PATH = os.path.join(data_path, "input_vector_sample")
INFO_INPUT_EMBEDDINGS_PATH = os.path.join(data_path, "info_input_embeddings.npy")
DEFAULT_CONFIG_FILE_PATH = os.path.join(resources_path, "youtube_deep_recall_model.cfg")
conf = configparser.ConfigParser()


class ModelConfiguration:
    def __init__(self):
        self.feature_index_type_map = {}
        self.attribute_dim_map = {}
        # 对指定特征设置维度
        self.feature_embedding_dim_map = {}
        self.from_config_file(DEFAULT_CONFIG_FILE_PATH)

    def from_config_file(self, file_path):
        conf.read(file_path)
        for option in conf.options("model_hyper_parameters"):
            exec("self." + option + " = " + conf.get("model_hyper_parameters", option))
        for option in conf.options("feature_index_type"):
            self.feature_index_type_map[option] = eval(conf.get("feature_index_type", option))
        for option in conf.options("attribute_dim"):
            self.attribute_dim_map[option] = conf.getint("attribute_dim", option)

    def from_command_line_arguments(self):
        FLAGS = flags.FLAGS
        if FLAGS.config:
            self.from_config_file(FLAGS.config)
        for option in ["epoch_num", "batch_size", "learn_rate"]:
            value = eval("FLAGS." + option)
            if value:
                exec("self." + option + " = " + value)

    def __str__(self):
        string = ""
        for option, argument in self.__dict__.items():
            string += ("--{} {}\n".format(option, argument))
        return string


if __name__ == "__main__":
    m = ModelConfiguration()
    print(m)
