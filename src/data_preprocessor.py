# coding=utf-8
"""
用于对原始数据进行处理
"""

import numpy as np
from src import config


def to_info_embedding_matrix(file_path):
    info_embedding_list = []
    for line in open(file_path):
        info_embedding_list.append(line.strip("\r\n").split(" "))
    return np.array(info_embedding_list, dtype='float32')


def to_instance(string):
    # string = "1\t0\t612\t0\t0\t0\t0\t0\t4\t\t\t978970\t1\t1"
    string = "1\t0\t612\t0\t0\t0\t0\t0\t4\t0\t0\t978970\t1\t1"
    l = []
    for feature, value in zip(config.FEATURE_LIST, string.strip("\r\n").split("\t")):
        if feature in config.SINGLE_FEATURE_LIST or feature in config.TARGET_FEATURE_LIST:
            l.append(value)
        elif feature in config.UNORDERED_FEATURE_LIST or feature in config.ORDERED_FEATURE_LIST:
            l.append(value.split(","))
        else:
            print("这是不可能的")
    return np.array(l, dtype=np.int32)


if __name__ == "__main__":
    # 将info_embedding文本文件保存为npy格式 压缩为20%
    np.save(config.INFO_INPUT_EMBEDDINGS_PATH, to_info_embedding_matrix(config.INFO_INPUT_EMBEDDINGS_TEXT_FILE_PATH))
