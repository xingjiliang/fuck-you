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
        info_input_embeddings = np.load(config.INFO_INPUT_EMBEDDINGS_PATH)

        self.model_config = model_config

        self.current_time = tf.placeholder(tf.int32, [None], name="current_time")
        self.slot = tf.placeholder(tf.int32, [None], name="slot")
        self.user_age_split = tf.placeholder(tf.int32, [None], name="user_age_split")
        self.user_gender = tf.placeholder(tf.int32, [None], name="user_gender")
        self.user_worked_years = tf.placeholder(tf.int32, [None], name="user_worked_years")
        self.os = tf.placeholder(tf.int32, [None], name="os")
        self.user_clicked_info_sequence = tf.placeholder(tf.int32, [None, None], name="user_clicked_info_list")
        self.user_delivered_info_sequence = tf.placeholder(tf.int32, [None, None], name="user_delivered_info_list")
        self.y_true = tf.placeholder(tf.int32, [None], name="y_true")

        self.info_input_embeddings = tf.get_variable(initializer=info_input_embeddings, name='info_input_embeddings')
        self.current_time_embeddings = tf.get_variable("current_time_embeddings", [3, config.default_embedding_size],
                                                       tf.float32)
        self.slot_embeddings = tf.get_variable("slot_embeddings", [3, config.default_embedding_size], tf.float32)
        self.user_age_split_embeddings = tf.get_variable("user_age_split_embeddings",
                                                         [3, config.default_embedding_size], tf.float32)
        self.user_gender_embeddings = tf.get_variable("user_gender_embeddings", [2, config.default_embedding_size],
                                                      tf.float32)
        self.user_delivered_info_sequence_embeddings = tf.nn.embedding_lookup(info_input_embeddings,
                                                                              self.user_delivered_info_sequence)
        self.reduced_user_delivered_info_sequence_embeddings = \
            tf.reduce_sum(self.user_delivered_info_sequence_embeddings, 1,
                          name="reduced_user_delivered_info_sequence_embeddings")

        self.concatenated_embedding = tf.concat([self.current_time_embeddings, self.user_gender_embeddings,
                                                 self.reduced_user_delivered_info_sequence_embeddings], 1,
                                                "so_called_user_embedding")

        temp_user_embedding = self.concatenated_embedding
        self.hidden_layer_list = []
        pre_layer_size = self.concatenated_embedding.shape[1]
        for i, hidden_layer_size in enumerate([128, 64]):
            # todo 可以添加偏置项

            temp_hidden_layer = tf.get_variable("hidden_layer_{}".format(i),
                                                [pre_layer_size, hidden_layer_size],
                                                tf.float32)
            temp_hidden_bias = tf.get_variable("hidden_layer_bias_{}".format(i), [hidden_layer_size, 1], tf.float32)
            temp_user_embedding = tf.relu(tf.matmul(temp_user_embedding, temp_hidden_layer))
            self.hidden_layer_list.append(temp_hidden_layer)
        self.so_called_user_embedding = temp_user_embedding

        # 注意在这个地方的尺寸
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
        self.opt = tf.train.AdamOptimizer(model_config.learn_rate)
        self.optimizer = self.opt.minimize(self.loss, global_step=self.global_step)
