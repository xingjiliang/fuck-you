# coding=utf-8
import tensorflow as tf

import config
import data_utils


class Model:
    """
    wide部分:交叉特征、部分离散特征
    deep部分:连续特征、离散特征embedding
    """

    def __init__(self, global_config, sample, info_input_embeddings, is_training):
        self.global_config = global_config

        self.feature_value_map = {}

        self.wide_part_feature_list = []
        self.deep_part_feature_list = []
        input_space_map = global_config.feature_config['input_space']
        for i, input_feature in enumerate(input_space_map):
            input_feature_attribute_map = input_space_map[input_feature]
            form = input_feature_attribute_map[config.INPUT_FEATURE_FORM]
            feature_type = input_feature_attribute_map[config.INPUT_FEATURE_TYPE]
            if form == 'cross' and feature_type == 'discrete':
                self.wide_part_feature_list.append(tf.expand_dims(sample[i], 1))
            elif form == 'single' and feature_type == 'continuous':
                if config.INPUT_FEATURE_OPS not in input_space_map[input_feature]:
                    self.deep_part_feature_list.append(tf.expand_dims(sample[i], 1))
                else:
                    self.deep_part_feature_list.append(tf.expand_dims(
                        data_utils.tensor_ops.ops[input_space_map[input_feature][config.INPUT_FEATURE_OPS]] \
                            (sample[i], input_space_map[input_feature][config.FUNC_PARAMS])), 1)
            else:
                self.labels = self.feature_value_map[input_feature]

        self.wide_part_vector = tf.cast(tf.concat(self.wide_part_feature_list, 1), tf.float32)
        self.wide_part_weights = tf.get_variable("wide_part_weights", [self.wide_part_vector.shape[1], 1], tf.float32,
                                                 tf.random_normal_initializer)
        self.deep_part_vector = tf.concat(self.deep_part_feature_list, 1)
        temp_hidden_vector = self.deep_part_vector
        pre_layer_size = temp_hidden_vector.shape[1]
        self.hidden_vector_list = []
        for i, hidden_layer_size in enumerate(global_config.full_connection_layer_list):
            temp_hidden_layer = tf.get_variable("hidden_layer_{}".format(i),
                                                [pre_layer_size, hidden_layer_size],
                                                tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
            temp_hidden_layer_bias = tf.get_variable("hidden_layer_bias_{}".format(i),
                                                     [1, hidden_layer_size],
                                                     tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer())
            if hidden_layer_size == 1:
                temp_hidden_vector = tf.concat([temp_hidden_vector, self.wide_part_vector], 1)
                temp_hidden_layer = tf.concat([temp_hidden_layer, self.wide_part_weights], 0)
            temp_vector = tf.add(tf.matmul(temp_hidden_vector, temp_hidden_layer), temp_hidden_layer_bias)
            temp_hidden_vector = tf.nn.relu(temp_vector) if hidden_layer_size != 1 else temp_vector
            pre_layer_size = hidden_layer_size
            self.hidden_vector_list.append(temp_hidden_vector)
        # 这里直接reduce_sum即可
        self.logits = temp_hidden_vector
        self.predictions = tf.where(self.logits < 0, tf.zeros_like(self.logits), tf.ones_like(self.logits))
        # 输入的logits应未被sigmoid归一化到(0,1)区间内
        sigmoid_cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.expand_dims(tf.cast(self.labels, tf.float32), -1), logits=self.logits,
        )
        self.loss = tf.reduce_mean(sigmoid_cross_entropy_loss)
