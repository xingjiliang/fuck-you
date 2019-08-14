# coding=utf-8
import tensorflow as tf


class Model:
    """
    未来或许可以实现wide&deep
    wide部分:交叉特征、部分离散特征
    deep部分:连续特征、离散特征embedding
    """
    def __init__(self, model_config, sample, info_input_embeddings, is_training):
        self.model_config = model_config

        self.feature_value_map = {}
        for i, feature in enumerate(model_config.feature_index_type_map):
            self.feature_value_map[feature] = sample[i]
        self.feature_embeddings_map = dict()
        for attribute in model_config.attribute_dim_map:
            self.feature_embeddings_map[attribute] = tf.get_variable(attribute + "_embeddings",
                                                                     [model_config.attribute_dim_map[attribute],
                                                                      model_config.default_attribute_embedding_size],
                                                                     tf.float32,
                                                                     initializer=tf.contrib.layers.xavier_initializer()
                                                                     )
        self.attribute_embeddings_list = []
        for feature in model_config.feature_index_type_map:
            index, feature_nature, feature_type, attribute = model_config.feature_index_type_map[feature]
            if feature_nature == "label":
                continue
            tensor = tf.nn.embedding_lookup(self.feature_embeddings_map[attribute],
                                            self.feature_value_map[feature])
                                            # 'div' if attribute == "info_input_embeddings" else 'mod')
            self.attribute_embeddings_list.append(tensor)

        self.concatenated_embeddings = tf.concat(self.attribute_embeddings_list, 1, "concatenated_embeddings")
        temp_hidden_vector = self.concatenated_embeddings
        pre_layer_size = self.temp_hidden_vector.shape[1]
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
            pre_layer_size = hidden_layer_size
            self.hidden_vector_list.append(temp_hidden_vector)
        self.logits = temp_hidden_vector
        sigmoid_cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.expand_dims(self.feature_value_map["delivery"], -1), logits=self.logits,
        )
        self.loss = tf.reduce_mean(sigmoid_cross_entropy_loss)
