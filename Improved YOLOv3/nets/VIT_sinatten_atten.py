from torch import nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        #   dim 输入特征的维度数

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

        x = (attn @ v).reshape(B, N, C)
        # x = (attn @ v).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class All_attention1(nn.Module):
    def __init__(self, dim):
        super(All_attention1, self).__init__()

        self.dim = dim
        self.atten = Attention(dim)
        self.sin_atten = Sin_Attention(dim)


    def forward(self, x):
        B, N, C = x.shape
        x1 = self.atten(x)
        x2 = self.sin_atten(x)
        x = (x1 + x2 + x).view(B, 52, 52, C)
        # x = x.reshape(B, self.dim ** -0.5, -1, C)
        return x

class All_attention2(nn.Module):
    def __init__(self, dim):
        super(All_attention2, self).__init__()

        self.dim = dim
        self.atten = Attention(dim)
        self.sin_atten = Sin_Attention(dim)


    def forward(self, x):
        B, N, C = x.shape
        x1 = self.atten(x)
        x2 = self.sin_atten(x)
        x = (x1 + x2 + x).view(B, 26, 26, C)
        # x = x.reshape(B, self.dim ** -0.5, -1, C)
        return x

class All_attention3(nn.Module):
    def __init__(self, dim):
        super(All_attention3, self).__init__()

        self.dim = dim
        self.atten = Attention(dim)
        self.sin_atten = Sin_Attention(dim)


    def forward(self, x):
        B, N, C = x.shape
        x1 = self.atten(x)
        x2 = self.sin_atten(x)
        x = (x1 + x2 + x).view(B, 13, 13, C)
        # x = x.reshape(B, self.dim ** -0.5, -1, C)
        return x