import torch
import torch.nn as nn
import copy
from torch.nn.init import xavier_normal_, constant_

class MLP(nn.Module):
    def __init__(self, configs, mode):
        super(MLP, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.period_len = configs.period_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        # 注意：这里的 seq_len 指的是切片后的数量 (num_slices)
        self.input = nn.Linear(self.seq_len, 512)
        # input_raw 指的是原始序列长度
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        # x: [B, num_slices, V] -> [B, V, num_slices]
        # x_raw: [B, L, V] -> [B, V, L]
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)

class Statistics_prediction(nn.Module):
    def __init__(self, configs):
        super(Statistics_prediction, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.period_len = configs.period_len
        self.channels = configs.enc_in
        self.station_type = configs.station_type

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

    def _build_model(self):
        args = copy.deepcopy(self.configs)
        # 修改 args 中的 seq_len 为切片数量，供 MLP 使用
        args.seq_len = self.configs.seq_len // self.period_len
        args.pred_len = self.configs.pred_len // self.period_len # 预测的切片数量
        args.c_out = self.configs.c_out # 确保 MLP 初始化正确
        
        self.model = MLP(args, mode='mean').float()
        self.model_std = MLP(args, mode='std').float()

    def normalize(self, input):
        if self.station_type == 'adaptive':
            bs, len_seq, dim = input.shape
            # 切分为 [B, Num_Slices, Period_Len, V]
            input_reshaped = input.reshape(bs, -1, self.period_len, dim)
            
            mean = torch.mean(input_reshaped, dim=-2, keepdim=True) # [B, Num, 1, V]
            std = torch.std(input_reshaped, dim=-2, keepdim=True)   # [B, Num, 1, V]
            norm_input = (input_reshaped - mean) / (std + self.epsilon)
            
            # 计算全局均值用于残差学习
            mean_all = torch.mean(input, dim=1, keepdim=True) # [B, 1, V]
            
            # 预测未来统计量
            outputs_mean = self.model(mean.squeeze(2) - mean_all, input - mean_all) * self.weight[0] + mean_all * self.weight[1]
            outputs_std = self.model_std(std.squeeze(2), input)

            outputs = torch.cat([outputs_mean, outputs_std], dim=-1) # [B, Pred_Slices, 2*V]

            return norm_input.reshape(bs, len_seq, dim), outputs

        else:
            return input, None

    def de_normalize(self, input, station_pred):
        if self.station_type == 'adaptive':
            bs, len_seq, dim = input.shape
            # input 是归一化后的预测结果
            input_reshaped = input.reshape(bs, -1, self.period_len, dim)
            
            # 解析预测出的统计量
            # station_pred shape: [B, Pred_Slices, 2*V]
            # 前半部分是 mean，后半部分是 std
            pred_mean = station_pred[:, :, :dim].unsqueeze(2)
            pred_std = station_pred[:, :, dim:].unsqueeze(2)
            
            output = input_reshaped * (pred_std + self.epsilon) + pred_mean
            return output.reshape(bs, len_seq, dim)
        else:
            return input
        
