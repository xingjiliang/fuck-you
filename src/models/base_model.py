# coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn_impl import _compute_sampled_logits

from src import config


class Model:
    def __init__(self, model_config):
        """
        定义模型结构,并初始化参数
        :param model_config:
        """
        self.model_config = model_config

        self.feature_placeholder_map = {}
        model_config = config.ModelConfiguration()
        for feature in model_config.feature_index_type_map:
            index, feature_nature, feature_type, attribute = model_config.feature_index_type_map[feature]
            self.feature_placeholder_map[feature] = \
                tf.placeholder(tf.int32 if "discrete" == feature_type else tf.float32,
                               [None] if feature_nature in ["single", "label"] else [None, None],
                               name=feature)

        info_input_embeddings = np.load(config.INFO_INPUT_EMBEDDINGS_PATH)
        self.feature_embeddings_map = {
            "info_input_embeddings": tf.get_variable(initializer=info_input_embeddings, name='info_input_embeddings')}
        for attribute in model_config.attribute_dim_map:
            self.feature_embeddings_map[attribute] = tf.get_variable(attribute + "_embeddings",
                                                                     [model_config.attribute_dim_map[attribute],
                                                                      config.default_embedding_size],
                                                                     tf.float32)

        func_dict = {
            "single": self.tackle_single_feature,
            "multi": self.tackle_multi_feature,
            "sequence": self.tackle_sequence_feature
        }
        self.before_fcn_embeddings_list = []
        for feature in model_config.feature_index_type_map:
            index, feature_nature, feature_type, attribute = model_config.feature_index_type_map[feature]
            if feature_nature == "label":
                continue
            tensor = tf.nn.embedding_lookup(self.feature_embeddings_map[attribute],
                                            self.feature_placeholder_map[feature])
            self.before_fcn_embeddings_list.append(func_dict[feature_type](tensor))

        self.so_called_raw_user_embedding = tf.concat(self.before_fcn_embeddings_list, 1, "so_called_raw_user_embedding")

        temp_hidden_vector = self.so_called_raw_user_embedding
        pre_layer_size = self.so_called_raw_user_embedding.shape[1]
        self.hidden_vector_list = []
        for i, hidden_layer_size in enumerate(model_config.full_connection_layer_list):
            temp_hidden_layer = tf.get_variable("hidden_layer_{}".format(i),
                                                [pre_layer_size, hidden_layer_size],
                                                tf.float32)
            temp_hidden_layer_bias = tf.get_variable("hidden_layer_bias_{}".format(i), [hidden_layer_size, 1], tf.float32)
            temp_hidden_vector = tf.relu(tf.add(tf.matmul(temp_hidden_vector, temp_hidden_layer), temp_hidden_layer_bias))
            self.hidden_vector_list.append(temp_hidden_vector)
        self.so_called_user_embedding = temp_hidden_vector

        # 注意在这个地方的尺寸 todo
        self.logits, self.labels = _compute_sampled_logits(
            weights=self.info_output_embeddings,  # 是随机初始化还是？ [input_embedding_size, 词典词数量]
            biases=tf.get_variable("classes_bias", model_config.info_size, tf.float32),  # [词典词数量]
            labels=tf.expand_dims(self.y_true, -1),  # [batch_size, true_size]
            inputs=self.so_called_user_embedding,  # [batch_size, input_embedding_size]
            num_sampled=10,  # 负采样数量
            num_classes=model_config.info_size,  # 词典词数量
            num_true=1,
            partition_strategy='div',
            name="nce_loss"
        )

        sigmoid_cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
                                                                             logits=self.logits,
                                                                             name="sigmoid_cross_entropy_loss")
        self.loss = tf.reduce_mean(tf.reduce_sum(sigmoid_cross_entropy_loss, 1))
        self.optimizer = tf.train.AdamOptimizer(model_config.learn_rate)
        self.optimize_operation = self.opt.minimize(self.loss, global_step=self.global_step)

    def tackle_single_feature(self, tensor):
        return tensor

    def tackle_sequence_feature(self, sequence_tensor):
        """
        输入的tensor的shape应当为[Batch, sequence_size, emb_size]
        :return:
        """
        return tf.reduce_sum(sequence_tensor, 1)

    def tackle_multi_feature(self, multi_tensor):
        """
        输入的tensor的shape应当为[Batch, list_size, emb_size]
        :return:
        """
        return tf.reduce_sum(multi_tensor, 1)

    def feature_to_attribute(self, feature):
        pass
