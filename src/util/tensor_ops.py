import tensorflow as tf


def min_max_scaling(x, func_params):
    """
    注意max_value和min_value不能相等
    :param x:
    :param min_value:
    :param max_value:
    :return:
    """
    max_value = func_params['max_value']
    min_value = func_params['min_value']
    return tf.divide(tf.subtract(x, min_value), tf.subtract(max_value, min_value))


ops = {
    "min_max_scaling": min_max_scaling
}
