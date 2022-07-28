import math
from collections import OrderedDict
from functools import partial
from nets.Single_attention import SingleVisionTransformer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sin_Attention(nn.Module):
    def __init__(self, dim, qkv_bias=False, mlp_ratio=4., attn_drop=0., proj_drop=0.):
        super().__init__()
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj = nn.Linear(dim, int(dim * mlp_ratio))
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        # x = (attn @ v).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        #   使用reshape将分割的不同部位再重变形，前期分割还是一个数据结构整体，所以可以使用reshape
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_All(nn.Module):
    def __init__(self, dim):
        super(Attention_All, self).__init__()
        self.sin_atten = Sin_Attention(dim)
        self.attn = Attention(dim)

    def forward(self, x):
        x1 = self.sin_atten(x)
        x2 = self.attn(x)
        x = x + x1 + x2
        return x