import torch
import torch.nn as nn
import math
class Weight_MSE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,y_true,y_pred):
        num = len(y_true[y_true>0])
        mask_result = torch.where(y_true==0,0,1)
        y_pred = torch.mul(y_pred,mask_result)
        weights = torch.clamp(y_true, math.log10(1 + 0.1 / 140), 1)
        return torch.sum(torch.matmul(weights,torch.pow((y_pred-y_true),2))) / num

class Weight_MAE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,y_true,y_pred):
        num = len(y_true[y_true > 0])
        mask_result = torch.where(y_true == 0, 0, 1)
        y_pred = torch.mul(y_pred, mask_result)
        weights = torch.clamp(y_true, math.log10(1 + 0.1 / 140), 1)
        return torch.sum(torch.matmul(weights,torch.abs(y_pred-y_true))) / num

class weight_mae(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,y_true,y_pred):
        num = len(y_true[y_true > 0])
        weights = torch.clamp(y_true, math.log10(1 + 0.1 / 140), 1)
        return torch.sum(torch.matmul(weights,torch.abs(y_pred-y_true))) / num
    
class weight_mse(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,y_true,y_pred):
        weights = torch.clamp(y_true, math.log10(1 + 0.1 / 140), 1)
        return torch.mean(torch.matmul(weights, torch.pow((y_pred - y_true), 2)))



