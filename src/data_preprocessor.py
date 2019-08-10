# coding=utf-8
"""
用于对原始数据进行处理
"""

import numpy as np
import tensorflow as tf

from src import config
from src.main import model_config


class DataSetProvider:
    def __init__(self, model_config):
        self.model_config = model_config

    def get_text_line_data_set(self, data_set_path):
        text_line_dataset = tf.data.TextLineDataset(data_set_path)


def to_info_embedding_matrix(file_path):
    info_embedding_list = []
    for line in open(file_path):
        info_embedding_list.append(line.strip("\r\n").split(" "))
    return np.array(info_embedding_list, dtype='float32')


def to_instance(line_tensor):
    split_line_tensor = tf.string_split([tf.string_strip(line_tensor)], "\t", False).values
    instance = []
    for feature in model_config.feature_index_type_map:
        index, feature_nature, feature_type, attribute = model_config.feature_index_type_map[feature]
        instance.append(tf.string_to_number(split_line_tensor[index], tf.int32 if feature_type == "discrete" else tf.float32)
                        if feature_nature == "single" or feature_nature == "label"
                        else tf.string_to_number(tf.string_split([tf.string_strip(split_line_tensor[index])], ",").values, tf.int32)
                        )
    return instance


if __name__ == "__main__":
    # 将info_embedding文本文件保存为npy格式 压缩为20%
    np.save(config.INFO_INPUT_EMBEDDINGS_PATH, to_info_embedding_matrix(config.INFO_INPUT_EMBEDDINGS_TEXT_FILE_PATH))
