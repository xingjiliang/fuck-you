# coding=utf-8
import tensorflow as tf

import config


class Model:
    """
    wide部分:交叉特征、部分离散特征
    deep部分:连续特征、离散特征embedding
    """

    def __init__(self, global_config, sample, info_input_embeddings, is_training):
        self.global_config = global_config

        self.feature_value_map = {}
        input_space_map = global_config.feature_config['input_space']
        for i, feature in enumerate(input_space_map):
            self.feature_value_map[feature] = sample[i]

        feature_space_map = global_config.feature_config['feature_space']
        self.feature_embeddings_map = {}
        for feature_space in feature_space_map:
            feature_space_attribute_map = feature_space_map[feature_space]
            initializer = tf.contrib.layers.xavier_initializer()
            exec('initializer = ' + feature_space_attribute_map[config.INITIALIZER] + '()')
            self.feature_embeddings_map[feature_space] = tf.get_variable(name=feature_space + "_embeddings",
                                                                         shape=[feature_space_attribute_map[
                                                                              config.INPUT_FEATURE_DIM],
                                                                          feature_space_attribute_map[
                                                                              config.EMBEDDING_DIM]],
                                                                         dtype=tf.float32,
                                                                         initializer=initializer
                                                                         )

        self.discrete_feature_embeddings_list = []
        for input_feature in input_space_map:
            input_feature_attribute_map = input_space_map[input_feature]
            form = input_feature_attribute_map[config.INPUT_FEATURE_FORM]
            feature_space = input_feature_attribute_map[config.FEATURE_SPACE]
            if form == "label":
                self.labels = self.feature_value_map[input_feature]
                continue
            tensor = tf.nn.embedding_lookup(self.feature_embeddings_map[feature_space],
                                            self.feature_value_map[input_feature])
            # 'div' if attribute == "info_input_embeddings" else 'mod')
            if form == 'single':
                self.discrete_feature_embeddings_list.append(tensor)
            if form == 'cross':
                self.final_vector_list.append(tensor)  # 这里tensor必须是[B,1]的shape才能成功

        self.concatenated_embeddings = tf.concat(self.discrete_feature_embeddings_list, 1, "concatenated_embeddings")
        temp_hidden_vector = self.concatenated_embeddings
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
            temp_vector = tf.add(tf.matmul(temp_hidden_vector, temp_hidden_layer), temp_hidden_layer_bias)
            temp_hidden_vector = tf.nn.relu(temp_vector)
            pre_layer_size = hidden_layer_size
            self.hidden_vector_list.append(temp_hidden_vector)
        # 这里直接reduce_sum即可
        self.logits = tf.reduce_sum(self.final_vector_list.append(temp_hidden_vector), 0)
        self.predictions = tf.where(self.logits < 0, tf.zeros_like(self.logits), tf.ones_like(self.logits))
        # 输入的logits应未被sigmoid归一化到(0,1)区间内
        sigmoid_cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.expand_dims(tf.cast(self.labels, tf.float32), -1), logits=self.logits,
        )
        self.loss = tf.reduce_mean(sigmoid_cross_entropy_loss)
