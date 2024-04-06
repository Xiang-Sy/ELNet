import logging
import math
from pathlib import Path
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import torch
from torch import nn
from models.common import *
from utils.torch_utils import model_info, initialize_weights


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(Conv(x, self.no * self.na, 1, p=0, act=False, bias=True, bn=False) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class LSAHEAD(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False
    
    def __init__(self, nc=80, anchors=(), size=(), ch=()):  # detection layer
        super(LSAHEAD, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.lsa = nn.ModuleList(LSA(size[i], ch[i]) for i in range(len(ch)))
        self.po = nn.ModuleList(Conv(x, 32, 1, 1, act=nn.ReLU()) for x in ch)
        self.pc = nn.ModuleList(Conv(x, 32, 1, 1, act=nn.ReLU()) for x in ch)
        self.mo = nn.ModuleList(Conv(32, 5 * self.na, 3, 1, act=False, bias=True, bn=False) for x in ch)
        self.mc = nn.ModuleList(Conv(32, self.nc * self.na, 1, 1, act=False, bias=True, bn=False) for x in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            sw = self.lsa[i](x[i]) + 1
            wx = x[i] * sw
            o = self.mo[i](self.po[i](wx))
            c = self.mc[i](self.pc[i](wx))
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = torch.cat([o.view(bs, self.na, 5, ny, nx).permute(0, 1, 3, 4, 2), c.view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2)], dim=-1).contiguous()
            
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg, ch=3, nc=None, anchors=None, imgsz=640):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        self.yaml['nc'] = nc  # override yaml value
        self.yaml['anchors'] = anchors if anchors is not None else self.yaml['anchors'] # override yaml value
        self.stride = torch.tensor([8., 16., 32.])
        self.model, self.save = parse_model(deepcopy(self.yaml), [int((imgsz[0] / x).item()) for x in self.stride], ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, LSAHEAD)):
            m.stride = self.stride
            m.anchors /= m.stride.view(-1, 1, 1)
            self._initialize_biases()  # only run once
        
        # Init weights, biases
        initialize_weights(self)
        self.info(img_size=imgsz)
        logger.info('')
    
    def forward(self, x):
        y = [] # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            
            y.append(x if m.i in self.save else None)  # save output
        
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]
        if isinstance(m, LSAHEAD):
            for mo, mc, s in zip(m.mo, m.mc, m.stride):  # from
                bo = mo.conv.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                bc = mc.conv.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                bo.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                bc.data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mo.conv.bias = torch.nn.Parameter(bo.view(-1), requires_grad=True)
                mc.conv.bias = torch.nn.Parameter(bc.view(-1), requires_grad=True)
        
        elif isinstance(m, Detect):
            for mi, s in zip(m.m, m.stride):  # from
                b = mi.conv.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        # m = self.model[0]
        # if isinstance(m, WTA):
        #     m.weighted[0].weight.data[:] = 0

    def info(self, verbose=False, img_size=[640, 640]):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, size, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc = d['anchors'], d['nc']

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['neck'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        
        if m in [Conv, C3, BNeck, EDFA]:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is WTA:
            c2 = ch[f]
            args = [c2, *args[:]]
        elif m in [Detect, LSAHEAD]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def attempt_load(weight, map_location=None):
    ckpt = torch.load(weight, map_location=map_location)  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].eval()  # FP32 model
    model.info()
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    return model  # return ensemble
