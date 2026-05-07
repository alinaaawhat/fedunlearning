import torch
import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def split_weight_name(name):
        if 'weight' in name or 'bias' in name:
            return ''.join(name.split('.')[:-1])
        return name

    def save_params(self):
        for param_name, param in self.named_parameters():
            if 'alpha' in param_name or 'beta' in param_name:
                continue
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name, param.data.clone())

    def compute_diff(self):
        diff_mean = {}
        for param_name, param in self.named_parameters():
            layer_name = self.split_weight_name(param_name)
            _buff_param_name = param_name.replace('.', '__')
            old_param = getattr(self, _buff_param_name, 0.0)
            diff = (param - old_param) ** 2
            diff = diff.sum()
            total_num = max(param.numel(), 1)
            diff /= total_num
            diff_mean[layer_name] = diff
        return diff_mean

    def remove_grad(self, name=''):
        for param_name, param in self.named_parameters():
            if name in param_name:
                param.requires_grad = False
