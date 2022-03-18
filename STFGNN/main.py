import os
import json
import torch

from data.dataset.stfgnn_dataset import STFGNNDataset
from model.STFGNN import STFGNN
from executor.multi_step_executor import MultiStepExecutor as STFGNNExecutor


config = {}
for filename in ["config/PEMS03.json", "config/STFGNN.json"]:
    with open(filename, "r") as f:
        _config = json.load(f)
        for key in _config:
            if key not in config:
                config[key] = _config[key]

dataset = STFGNNDataset(config)

train_data, valid_data, test_data = dataset.get_data()
data_feature = dataset.get_data_feature()

model_cache_file = 'cache/model_cache/PEMS03_STFGNN.m'

model = STFGNN(config, data_feature)

executor = STFGNNExecutor(config, model)


train = True #标识是否需要重新训练

if train or not os.path.exists(model_cache_file):
    executor.train(train_data, valid_data)
    executor.save_model(model_cache_file)
else:
    executor.load_model(model_cache_file)
# 评估，评估结果将会放在 cache/evaluate_cache 下
executor.evaluate(test_data)





