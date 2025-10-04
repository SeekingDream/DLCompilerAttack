import torch
import torch.nn as nn
import copy


class BiasLayer(nn.Module):
    def __init__(self, v_shape):
        super(BiasLayer, self).__init__()
        self.V = torch.zeros(v_shape)
        self.M = torch.ones_like(self.V)
        self.reverse_M = torch.ones_like(self.V)
        self.index = None

    def forward(self, x):
        return self.M * (x - self.V)

    def set_v(self, V, M):
        self.V = V
        self.M = torch.ones_like(V)
        self.index = torch.where(V.reshape([-1]) != 0)[0]
        self.M.reshape([-1])[self.index] = M * 10

    def to(self, device):
        self.V = self.V.to(device)
        self.M = self.M.to(device)
        return self 


class MyModel(nn.Module):
    def __init__(self, m_1: nn.Module, act: nn.Module, m_2: nn.Module):
        super(MyModel, self).__init__()
        self.fp = m_1.fp
        self.m_1 = copy.deepcopy(m_1)
        self.act = act
        self.m_2 = copy.deepcopy(m_2)

        self.input_sizes = m_1.input_sizes
        self.input_types = m_1.input_types
        # self.class_num = m_1.class_num

    def init(self):
        self.input_sizes = self.m_1.input_sizes
        self.input_types = self.m_1.input_types
        # self.class_num = self.m_1.class_num

    def forward(self, x):
        m1_out = self.m_1(x)
        m1_act = self.act(m1_out.to(x.device))
        m2_out = self.m_2(m1_act.to(x.device))
        return [m2_out, m1_out]

    def to(self, device):
        self.m_2.input_bias.to(device)
        super(MyModel, self).to(device)
        return self

class WrapperModel(nn.Module):
    def __init__(self):
        super(WrapperModel, self).__init__()
        self.input_sizes = None
        self.model_data_name = None
        self.input_types = ['float32']
        self.fp = torch.float32