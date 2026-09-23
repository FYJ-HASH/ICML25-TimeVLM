from src.TimeVLM.sam_decomp_optimizer import SAMDecompOptimizer
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw

warnings.filterwarnings('ignore')

class Exp_Few_Shot_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Few_Shot_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if getattr(self.args, 'use_sam', False):
            model = self.model.module if hasattr(self.model, 'module') else self.model

            param_groups = [
                {"params": list(model.vlm_model.vision_model.parameters()) if hasattr(model.vlm_model, 'vision_model') else [],
                 "name": "modal1"
