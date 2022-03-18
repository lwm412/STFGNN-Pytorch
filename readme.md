### STFGNN-Pytorch

This is the pytorch re-implementation of the STFGNN model described in https://arxiv.org/abs/2012.09641.

#### Quick start

Put your data in STFGNN/raw_data. 

For example, if you want to run model on dataset PEMS03, put the file **adj_mx.pkl** and file **PEMS03.npz** in **STFGNN/raw_data/PEMS03/** .

Set appropriate value of parameter in **STFGNN/config/*.json**.

Run **python main.py** to run the model. 

