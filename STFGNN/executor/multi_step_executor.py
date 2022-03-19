import os
import time
import numpy as np
import torch
import math
import time
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
import tqdm
from torch.utils.tensorboard import SummaryWriter
from executor.utils import get_train_loss
from utils.Optim import Optim
from evaluator.evaluator import Evaluator
from utils.utils import ensure_dir

from model import loss
from functools import partial


class MultiStepExecutor(object):
    def __init__(self, config, model):
        self.config = config
        self.evaluator = Evaluator(config)

        _device = self.config.get('device', torch.device('cpu'))
        self.device = torch.device(_device)
        self.model = model.to(self.device)

        self.cache_dir = 'cache/model_cache'
        self.evaluate_res_dir = 'cache/evaluate_cache'
        self.summary_writer_dir = 'log/runs'
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._logger.info(self.model)

        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))

        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.train_loss = self.config.get("train_loss", "masked_mae")
        self.criterion = get_train_loss(self.train_loss) 

        self.cuda = self.config.get("cuda", True)
        self.best_val = 10000000
        self.optim = Optim(
            model.parameters(), self.config
        )
        self.epochs = self.config.get("epochs", 100)
        self.scaler = self.model.scaler
        self.num_batches = self.model.num_batches
        self.num_nodes = self.config.get("num_nodes", 0)
        self.batch_size = self.config.get("batch_size", 64)
        self.patience = self.config.get("patience", 20)
        self.lr_decay = self.config.get("lr_decay", False)
        self.mask = self.config.get("mask", True)


    def train(self, train_data, valid_data):
        print("begin training")
        wait = 0
        batches_seen = self.num_batches * 0
        

        for epoch in tqdm.tqdm(range(1, self.epochs + 1)):
            epoch_start_time = time.time()
            train_loss = []
            train_data.shuffle()

            for iter, (x,y) in enumerate(train_data.get_iterator()):
                self.model.train()
                self.model.zero_grad() 
                trainx = torch.Tensor(x).to(self.device)  # [batch_size, window, num_nodes, dim]
                trainy = torch.Tensor(y).to(self.device)  # [batch_size, horizon, num_nodes, dim]
                output = self.model(trainx)
                loss = self.criterion(self.scaler.inverse_transform(output), 
                    self.scaler.inverse_transform(trainy))
                
                loss.backward()
                self.optim.step()
                train_loss.append(loss.item())
                
            
            if self.lr_decay:
                self.optim.lr_scheduler.step()

            valid_loss = []
            valid_mape = []
            valid_rmse = []
            valid_pcc = []
            for iter, (x, y) in enumerate(valid_data.get_iterator()):
                self.model.eval()
                valx = torch.Tensor(x).to(self.device)
                valy = torch.Tensor(y).to(self.device)
                with torch.no_grad():
                    output = self.model(valx)
                score = self.evaluator.evaluate(self.scaler.inverse_transform(output), \
                    self.scaler.inverse_transform(valy))
                if self.mask:
                    vloss = score["masked_MAE"]["all"]
                else:
                    vloss = score["MAE"]["all"]
                    
                valid_loss.append(vloss)
            

            mtrain_loss = np.mean(train_loss)

            mvalid_loss = np.mean(valid_loss)

            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid mae {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), mtrain_loss, \
                        mvalid_loss))

            if mvalid_loss < self.best_val:
                self.best_val = mvalid_loss
                wait = 0
                self.best_val = mvalid_loss
                self.best_model = self.model
            else:
                wait += 1

            if wait >= self.patience:
                print('early stop at epoch: {:04d}'.format(epoch))
                break
        
        self.model = self.best_model


    def evaluate(self, test_data):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        outputs = []
        realy = []
        seq_len = test_data.seq_len  #test_data["y_test"]
        self.model.eval()
        for iter, (x, y) in enumerate(test_data.get_iterator()):
            testx = torch.Tensor(x).to(self.device)
            testy = torch.Tensor(y).to(self.device)
            with torch.no_grad():
                # self.evaluator.clear()
                pred = self.model(testx)
                outputs.append(pred) 
                realy.append(testy)
        realy = torch.cat(realy, dim=0)
        yhat = torch.cat(outputs, dim=0)

        realy = realy[:seq_len, ...]
        yhat = yhat[:seq_len, ...]

        realy = self.scaler.inverse_transform(realy)
        preds = self.scaler.inverse_transform(yhat)

        res_scores = self.evaluator.evaluate(preds, realy)
        for _index in res_scores.keys():
            print(_index, " :")
            step_dict = res_scores[_index]
            for j, k in step_dict.items():
                print(j, " : ", k.item())
        
        

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save(self.model.state_dict(), cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
