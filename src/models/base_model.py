# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops.nn_impl import _compute_sampled_logits

from src import config
from src import data_utils


class Model:
    def __init__(self, model_config, sample, info_input_embeddings, is_training):
        """
        定义模型结构,并初始化参数
        :param model_config:
        """
        self.model_config = model_config

        self.feature_value_map = {}
        for i, feature in enumerate(model_config.feature_index_type_map):
            self.feature_value_map[feature] = sample[i]

        def _get_tensor_list(name, array, shard_num):
            tensor_list = []
            array_list = data_utils.shard_array(array, shard_num)
            for i, array in enumerate(array_list):
                tensor_list.append(
                    tf.get_variable('{}_sharded{}'.format(name, i), initializer=array, caching_device='/cpu:0')
                )
            return tensor_list

        # cpu_ts = tf.get_variable("cpu_arr", initializer=arr, dtype=tf.float32, caching_device='/cpu:0')
        self.feature_embeddings_map = {
            "info_input_embeddings": _get_tensor_list('info_input_embeddings', info_input_embeddings, 3),
            "info_output_embeddings": tf.get_variable('info_output_embeddings', info_input_embeddings.shape, tf.float32,
                                                      initializer=tf.random_normal_initializer(),
                                                      caching_device='/cpu:0',
                                                      # TODO: 目前为省时直接指定一个固定值, 15,000,000 * 50dim * 8byte
                                                      # FIXME: 当修改了这里的partitioner时，注意下面的embedding_lookup 和nce_loss都需要修改为'div'
                                                      # partitioner=tf.fixed_size_partitioner(3)
                                                      )}
        for attribute in model_config.attribute_dim_map:
            self.feature_embeddings_map[attribute] = tf.get_variable(attribute + "_embeddings",
                                                                     [model_config.attribute_dim_map[attribute],
                                                                      config.default_embedding_size],
                                                                     tf.float32,
                                                                     initializer=tf.contrib.layers.xavier_initializer())

        func_dict = {
            "single": lambda x: x,
            "multi": lambda x: tf.reduce_sum(x, 1),
            "sequence": lambda x: tf.reduce_sum(x, 1)
        }
        self.before_fcn_embeddings_list = []
        for feature in model_config.feature_index_type_map:
            index, feature_nature, feature_type, attribute = model_config.feature_index_type_map[feature]
            if feature_nature == "label":
                continue
            tensor = tf.nn.embedding_lookup(self.feature_embeddings_map[attribute],
                                            self.feature_value_map[feature],
                                            'div' if attribute == "info_input_embeddings" else 'mod')
            self.before_fcn_embeddings_list.append(func_dict[feature_nature](tensor))

        self.so_called_raw_user_embedding = tf.concat(self.before_fcn_embeddings_list, 1, "so_called_raw_user_embedding")

        temp_hidden_vector = self.so_called_raw_user_embedding
        pre_layer_size = self.so_called_raw_user_embedding.shape[1]
        self.hidden_vector_list = []
        for i, hidden_layer_size in enumerate(model_config.full_connection_layer_list):
            temp_hidden_layer = tf.get_variable("hidden_layer_{}".format(i),
                                                [pre_layer_size, hidden_layer_size],
                                                tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
            temp_hidden_layer_bias = tf.get_variable("hidden_layer_bias_{}".format(i),
                                                     [1, hidden_layer_size],
                                                     tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer())
            temp_hidden_vector = tf.nn.relu(
                tf.add(tf.matmul(temp_hidden_vector, temp_hidden_layer), temp_hidden_layer_bias)
            )
            self.hidden_vector_list.append(temp_hidden_vector)
        self.so_called_user_embedding = temp_hidden_vector

        # 注意在这个地方的尺寸 todo so_called_user_embedding和info_output_embeddings需要一致
        self.logits, self.labels = _compute_sampled_logits(
            weights=self.feature_embeddings_map["info_output_embeddings"],  # 是随机初始化还是？ [input_embedding_size, 词典词数量]
            biases=tf.get_variable("nce_classes_bias", model_config.info_size, tf.float32,
                                   initializer=tf.contrib.layers.random_normal_initializer()),  # [词典词数量]
            inputs=self.so_called_user_embedding,  # [batch_size, input_embedding_size]
            labels=self.feature_value_map["info_id"],  # [batch_size, true_size]
            num_sampled=10,  # 负采样数量
            num_classes=model_config.info_size,  # 词典词数量
            num_true=1,
            partition_strategy='mod',  # 'div'
            name="nce_loss"
        )

        sigmoid_cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
                                                                             logits=self.logits,
                                                                             name="sigmoid_cross_entropy_loss")
        self.loss = tf.reduce_mean(tf.reduce_sum(sigmoid_cross_entropy_loss, 1))
        # self.loss = tf.reduce_mean(tf.nn.nce_loss(
        #     weights=self.feature_embeddings_map["info_output_embeddings"],
        #     biases=tf.get_variable("nce_classes_bias", model_config.info_size, tf.float32),
        #     inputs=self.so_called_user_embedding,  # [batch_size, input_embedding_size]
        #     labels=self.feature_value_map["info_id"],  # [batch_size, true_size]
        #     num_sampled=10,  # 负采样数量
        #     num_classes=model_config.info_size,  # 词典词数量
        #     num_true=1,
        #     partition_strategy='mod'
        # ), 1)
