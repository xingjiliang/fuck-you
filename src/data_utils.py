# coding=utf-8
"""
用于对原始数据进行处理
"""

import numpy as np
import tensorflow as tf

from src import config


def from_text_line_file(dataset_path, is_trainning, dataset_config):
    text_line_dataset = tf.data.TextLineDataset(dataset_path)
    padded_shapes_list = []
    for feature in dataset_config.feature_index_type_map:
        index, feature_nature, feature_type, attribute = dataset_config.feature_index_type_map[feature]
        # ([], [], [], [], [], [], [], [50], [50], [50], [50], [], [], [])
        padded_shapes_list.append([] if feature_nature == "single" or feature_nature == "label"
                         else [dataset_config.max_sequence_size])

    def to_instance(line_tensor):
        split_line_tensor = tf.string_split([tf.string_strip(line_tensor)], "\t", False).values
        instance = []
        for feature in dataset_config.feature_index_type_map:
            index, feature_nature, feature_type, attribute = dataset_config.feature_index_type_map[feature]
            instance.append(tf.string_to_number(split_line_tensor[index],
                                                tf.int32 if feature_type == "discrete" else tf.float32
                                                )
                            if feature_nature == "single" or feature_nature == "label"
                            else tf.string_to_number(
                tf.string_split([tf.string_strip(split_line_tensor[index])], ",").values, tf.int32)
            )
        return instance

    if is_trainning:
        return text_line_dataset.map(to_instance).padded_batch(dataset_config.batch_size, tuple(padded_shapes_list)).\
            repeat(dataset_config.epoch_num).\
            shuffle(10000)
    else:
        return text_line_dataset.map(to_instance).padded_batch(10000, tuple(padded_shapes_list))


def shard_array(array, shard_num):
    array_row_num = array.shape[0]
    sharded_batch_size = array_row_num // shard_num
    mod = array_row_num % shard_num
    array_list = []
    start_idx = 0
    for i in range(shard_num):
        if mod > 0:
            array_list.append(array[start_idx: start_idx + sharded_batch_size + 1])
            start_idx += sharded_batch_size + 1
            mod = mod - 1
        else:
            array_list.append(array[start_idx: start_idx + sharded_batch_size])
            start_idx += sharded_batch_size
    return array_list


def to_info_embedding_array(file_path):
    info_embedding_list = []
    for line in open(file_path):
        info_embedding_list.append(line.strip("\r\n").split(" ")[1:])
    return np.array(info_embedding_list, dtype='float32')


if __name__ == "__main__":
    # 将info_embedding文本文件保存为npy格式 压缩为20%
    np.save(config.INFO_INPUT_EMBEDDINGS_PATH, to_info_embedding_array(config.INFO_INPUT_EMBEDDINGS_TEXT_FILE_PATH))
