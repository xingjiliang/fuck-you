# coding=utf-8
"""
用于对原始数据进行处理
"""
import os

import numpy as np
import tensorflow as tf

import config
import util.tensor_ops as tensor_ops


def from_text_line_file(dataset_config, is_trainning):
    dataset_path = dataset_config.train_dataset_path if is_trainning else dataset_config.test_dataset_path
    text_line_dataset = tf.data.TextLineDataset(dataset_path if os.path.isfile(dataset_path)
                                                else [os.path.join(dataset_path, file_path) for file_path in os.listdir(dataset_path)])

    def to_instance(line_tensor):
        split_line_tensor = tf.string_split([tf.string_strip(line_tensor)], "\t", False).values
        instance = []
        to_instance_input_feature_map = dataset_config.feature_config[config.INPUT_FEATURE_SPACE]
        for to_instance_feature in to_instance_input_feature_map:
            to_instance_feature_attribute_map = to_instance_input_feature_map[to_instance_feature]
            to_instance_feature_index = to_instance_feature_attribute_map[config.INPUT_FEATURE_INDEX]
            to_instance_feature_form = to_instance_feature_attribute_map[config.INPUT_FEATURE_FORM]
            to_instance_feature_type = to_instance_feature_attribute_map[config.INPUT_FEATURE_TYPE]
            instance.append(tf.string_to_number(split_line_tensor[to_instance_feature_index],
                                                tf.int32 if to_instance_feature_type == "discrete" else tf.float32
                                                )
                            if to_instance_feature_form == "single" or to_instance_feature_form == "label"
                            else tf.string_to_number(
                tf.string_split([tf.string_strip(split_line_tensor[to_instance_feature_index])], ",").values, tf.int32)
                            )
        return instance

    padded_shapes_list = []
    input_feature_map = dataset_config.feature_config[config.INPUT_FEATURE_SPACE]
    for feature in input_feature_map:
        feature_attribute_map = input_feature_map[feature]
        feature_form = feature_attribute_map[config.INPUT_FEATURE_FORM]
        # ([], [], [], [], [], [], [], [50], [50], [50], [50], [], [], [])
        padded_shapes_list.append([] if feature_form == "single" or feature_form == "label" or feature_form == 'cross'
                                  else [dataset_config.max_sequence_size])
    print(padded_shapes_list)
    print(text_line_dataset.map(to_instance))
    if is_trainning:
        return text_line_dataset.map(to_instance)\
            .shuffle(dataset_config.shuffle_buffer_size)\
            .padded_batch(dataset_config.batch_size, tuple(padded_shapes_list))\
            .repeat(dataset_config.epoch_num)

    else:
        return text_line_dataset.map(to_instance).padded_batch(dataset_config.batch_size, tuple(padded_shapes_list))


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


def min_max_scaling(x, min_value, max_value):
    return (x - min_value) / (max_value - min_value)


if __name__ == "__main__":
    # 将info_embedding文本文件保存为npy格式 压缩为20%
    np.save(config.INFO_INPUT_EMBEDDINGS_PATH, to_info_embedding_array(config.INFO_INPUT_EMBEDDINGS_TEXT_FILE_PATH))
