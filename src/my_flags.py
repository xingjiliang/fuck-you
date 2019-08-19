class MyFlags:
    def __init__(self):
        self.application_name = "demo"
        self.model_name = "fnn"
        self.train_dataset_path = "fnn_train.small"
        self.test_dataset_path = "fnn_test.small"
        self.epoch_num = 1
        self.batch_size = 10
        self.learn_rate = 1.0
        self.model_config_file = "im_click_rate_prediction_model.cfg"
        self.feature_config_file = "features.json"
