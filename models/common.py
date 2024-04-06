import torch
import torch.nn as nn
import ptwt
from einops import rearrange
from einops.layers.torch import Rearrange


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=None, bias=False, bn=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(c2) if bn is True else nn.Identity()
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        return self.act(self.bn(out))


class BNeck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(BNeck, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Conv(inp, inp, 3, stride, g=inp, act=nn.ReLU6()),
                # pw-linear
                Conv(inp, oup, 1, 1))
        else:
            self.conv = nn.Sequential(
                # pw
                Conv(inp, hidden_dim, 1, 1, act=nn.ReLU6()),
                # dw
                Conv(hidden_dim, hidden_dim, 3, stride, g=hidden_dim, act=nn.ReLU6()),
                # pw-linear
                Conv(hidden_dim, oup, 1, 1))

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=nn.ReLU())
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=nn.ReLU())
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=nn.ReLU())
        self.cv2 = Conv(c1, c_, 1, 1, act=nn.ReLU())
        self.cv3 = Conv(2 * c_, c2, 1, act=nn.ReLU())  # optional act=FReLU(c2)
        self.m = Bottleneck(c_, c_, shortcut, g, e=1.0)

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class WTA(nn.Module):
    def __init__(self, c, image_size, patch_size, h, hd):
        super(WTA, self).__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        self.c = c
        dim = 4 * self.c
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 4 * self.c * patch_size * patch_size
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim))
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        self.transformer = Transformer(dim, 2, h, hd, dim*8)

        self.weighted = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=(image_size//patch_size)),
            Conv(dim, dim, 3, 1, g=dim, act=nn.ReLU(), bn=False),
            nn.AvgPool2d(image_size//patch_size),
            nn.Sigmoid())
    
    def forward(self, x):
        dtype = x.dtype
        
        a, (b, c, d) = ptwt.wavedec2(x.float(), 'haar', 'zero', 1)
        z = torch.cat((a, b, c, d), 1).to(dtype)
        
        ti = self.transformer(self.to_patch_embedding(z))
        w = self.weighted(ti) * 2
        
        out = (z * w).float()
        return ptwt.waverec2([out[:,:self.c], (out[:,self.c:self.c*2], out[:,self.c*2:self.c*3], out[:,self.c*3:])], 'haar').to(dtype)


class EDFA(nn.Module):
    def __init__(self, c1, c2, l):
        super(EDFA, self).__init__()
        
        self.conv1 = Conv(c1, c1 // 2, act=nn.ReLU())
        self.conv2 = Conv(c1, c1 // 2, act=nn.ReLU())
        
        self.s1 = nn.Sequential(
            Rearrange('b c h w -> b w h c'),
            Conv(l, l, act=False),
            Rearrange('b w h c -> b h w c'),
            Conv(l, l, (1, 3), act=nn.ReLU()),
            Rearrange('b h w c -> b c h w'))
        
        self.s2 = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            Conv(l, l, act=False),
            Rearrange('b h w c -> b w h c'),
            Conv(l, l, (1, 3), act=nn.ReLU()),
            Rearrange('b w h c -> b c h w'))
        
        self.oc = Conv(c1, c2, act=nn.ReLU())

        self.res = Conv(c1, c2, act=nn.ReLU()) if c1 != c2 else nn.Identity()
    
    def forward(self, x):
        y1 = self.s1(self.conv1(x))
        y2 = self.s2(self.conv2(x))
        
        return self.oc(torch.cat([y1, y2], dim=1)) + self.res(x)


class LSA(nn.Module):
    def __init__(self, size, dim):
        super(LSA, self).__init__()
        
        self.conv = Conv(dim, 2, 1, act = nn.ReLU())
        
        self.ww = nn.Sequential(
            Conv(2, 1, (size, 5), p=(0,2), bn=False),
            Rearrange('b 1 1 s -> b s 1'),
            nn.Conv1d(size, size, 1, 1),
            nn.Sigmoid()
        )
        
        self.hw = nn.Sequential(
            Conv(2, 1, (5, size), p=(2,0), bn=False),
            Rearrange('b 1 s 1 -> b s 1'),
            nn.Conv1d(size, size, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        bs = x.shape[0]
        
        ww = self.ww(x)
        hw = self.hw(x)
        
        return torch.cat([torch.mm(hw[i],ww[i].t()).unsqueeze(0).unsqueeze(0) for i in range(bs)], dim=0)
