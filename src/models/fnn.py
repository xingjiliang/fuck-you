# coding=utf-8
import tensorflow as tf

import config
import data_utils


class Model:

    def __init__(self, global_config, sample, info_input_embeddings, is_training):
        self.global_config = global_config

        self.feature_value_map = {}
        input_space_map = global_config.feature_config['input_space']
        for i, input_feature in enumerate(input_space_map):
            if config.INPUT_FEATURE_OPS not in input_space_map[input_feature]:
                self.feature_value_map[input_feature] = sample[i]
            else:
                self.feature_value_map[input_feature] = \
                    data_utils.tensor_ops.ops[input_space_map[input_feature][config.INPUT_FEATURE_OPS]] \
                        (sample[i], input_space_map[input_feature][config.FUNC_PARAMS])
            # todo 未来换成列表
            # self.feature_value_map[input_feature] = sample[i] \
            #     if config.INPUT_FEATURE_OPS not in input_space_map[input_feature] \
            #     else data_utils.tensor_ops.ops[input_space_map[input_feature][config.INPUT_FEATURE_OPS]]

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

        self.feature_embeddings_list = []
        for input_feature in input_space_map:
            input_feature_attribute_map = input_space_map[input_feature]
            form = input_feature_attribute_map[config.INPUT_FEATURE_FORM]
            feature_type = input_feature_attribute_map[config.INPUT_FEATURE_TYPE]
            if feature_type == 'discrete':
                feature_space = input_feature_attribute_map[config.FEATURE_SPACE]
            if form == "label":
                self.labels = self.feature_value_map[input_feature]
                continue
            if feature_type == 'discrete':
                tensor = tf.nn.embedding_lookup(self.feature_embeddings_map[feature_space],
                                                self.feature_value_map[input_feature])
                # 'div' if attribute == "info_input_embeddings" else 'mod')
                self.feature_embeddings_list.append(tensor)
            elif feature_type == 'continuous':
                # todo 确认self.feature_value_map[input_feature]的shape是否为[B, 1]
                self.feature_embeddings_list.append(self.feature_value_map[input_feature])
        print(self.feature_embeddings_list)
        self.concatenated_embeddings = tf.concat(self.feature_embeddings_list, 1, "concatenated_embeddings")
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
            temp_hidden_vector = tf.nn.relu(temp_vector) if hidden_layer_size != 1 else temp_vector
            pre_layer_size = hidden_layer_size
            self.hidden_vector_list.append(temp_hidden_vector)
        self.logits = temp_hidden_vector
        self.predictions = tf.where(self.logits < 0, tf.zeros_like(self.logits), tf.ones_like(self.logits))
        # 输入的logits应未被sigmoid归一化到(0,1)区间内
        sigmoid_cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.expand_dims(tf.cast(self.labels, tf.float32), -1), logits=self.logits,
        )
        self.loss = tf.reduce_mean(sigmoid_cross_entropy_loss)
