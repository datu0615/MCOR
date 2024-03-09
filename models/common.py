# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized

from torch.nn import init, Sequential
from torch_geometric.nn import GCNConv 


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])
    

class Add_CS(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add_CS, self).__init__()
        self.arg = arg

    def forward(self, x):
        output = torch.add(x[0], x[1])
        permuted_channels = torch.randperm(output.size(1))
        shuffled_tensor = output[:, permuted_channels, :, :]
        return shuffled_tensor


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])

class Add2_2(nn.Module):
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        # x에 따라 다른 텐서 추가
        if self.index == 0:
            result = torch.add(x[0], x[1][0])
            # 배치별 채널 평균 계산
            channel_means = torch.mean(result, dim=1, keepdim=True)
            # 결과를 이미지로 저장
            self.save_images(channel_means, type='rgb')
        elif self.index == 1:
            result = torch.add(x[0], x[1][1])
            # 배치별 채널 평균 계산
            channel_means = torch.mean(result, dim=1, keepdim=True)
            # 결과를 이미지로 저장
            self.save_images(channel_means, type='ir')

        return channel_means

    def save_images(self, tensor, directory='/home/jb/docker/od/multispectral-object-detection/feature_map/', filename_prefix='feature_map', type='ir'):
        # tensor: (B, C, H, W) 형태의 텐서
        # 배치 내 각 샘플에 대해 반복
        for i, image in enumerate(tensor):
            # C = 1이므로, 첫 번째 채널을 선택
            print(image.shape)
            image = image.squeeze().detach()  # 차원 축소: (H, W)
            
            # PIL 이미지로 변환
            image = image.cpu().numpy()
            # image = Image.fromarray(image)
            
            image = Image.fromarray((image * 255))
            # 이미지 저장
            filename = f"{directory}/{filename_prefix}_{type}_{i}.png"
            image.save(filename)

class Add2_CS(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            output = torch.add(x[0], x[1][0])
            permuted_channels = torch.randperm(output.size(1))
            shuffled_tensor = output[:, permuted_channels, :, :]
            return shuffled_tensor
        elif self.index == 1:
            output = torch.add(x[0], x[1][1])
            permuted_channels = torch.randperm(output.size(1))
            shuffled_tensor = output[:, permuted_channels, :, :]
            return shuffled_tensor
        # return torch.add(x[0], x[1])


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out


class GraphGPT(nn.Module):
    def __init__(self, d_model, n_layer=8, vert_anchors=8, horz_anchors=8, embd_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        # Positional embedding parameter (learnable), for RGB and IR features
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # Graph convolution layers
        self.graph_layers = nn.ModuleList([
            GCNConv(d_model, d_model) for _ in range(n_layer)
        ])

        # Layer normalization and dropout
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, edge_index):
        rgb_fea, ir_fea = x  # Unpacking the input tuple

        # Check input dimensions
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # Average Pooling to reduce dimensionality
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # Flatten and concatenate features
        rgb_fea_flat = rgb_fea.view(bs, c, -1)
        ir_fea_flat = ir_fea.view(bs, c, -1)
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()

        # Apply graph convolution layers
        x = self.drop(self.pos_emb + token_embeddings)
        for layer in self.graph_layers:
            x = F.relu(layer(x, edge_index))

        # Apply layer normalization
        x = self.ln_f(x)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)

        # Split and reshape output for RGB and IR features
        rgb_fea_out = x[:, 0].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # Interpolate to original dimensions
        rgb_fea_out = F


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.sigmoid = nn.Sigmoid()
        #self.act=SiLU()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)
        
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) 
        #self.act=SiLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, c1,c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
    

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        # output = self.cv2(torch.cat(y, 1))
        # print(f'c2f ouput : {output.shape}')
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    

# def cosine_similarity_channel_sorting(rgb, ir, sa):
#     sa = torch.sigmoid(sa)
#     rsa = 1 - sa
#     # Normalize tensor1 and tensor2
#     sa_norm = F.normalize(sa, p=2, dim=(2,3), eps=1e-12)
#     rgb_norm = F.normalize(rgb, p=2, dim=(2,3), eps=1e-12)
#     ir_norm = F.normalize(ir, p=2, dim=(2,3), eps=1e-12)

#     # Compute cosine similarity
#     similarity_rgb = (sa_norm * rgb_norm).sum(dim=(2,3))
#     similarity_ir = (sa_norm * ir_norm).sum(dim=(2,3))

#     sorted_values_rgb, sorted_indices_rgb = torch.sort(similarity_rgb, dim=1, descending=False)
#     sorted_values_ir, sorted_indices_ir = torch.sort(similarity_ir, dim=1, descending=False)

#     rgb_sorted = torch.gather(rgb, 1, sorted_indices_rgb.unsqueeze(2).unsqueeze(3).expand(-1, -1, rgb.size(2), rgb.size(3)))
#     ir_sorted = torch.gather(ir, 1, sorted_indices_ir.unsqueeze(2).unsqueeze(3).expand(-1, -1, ir.size(2), ir.size(3)))

#     positive_mask_rgb, negative_mask_rgb = sorted_values_rgb > 0, sorted_values_rgb <= 0
#     positive_mask_ir, negative_mask_ir = sorted_values_ir > 0, sorted_values_ir <= 0

#     num_positive_channels_rgb, num_negative_channels_rgb = positive_mask_rgb.sum(dim=1), negative_mask_rgb.sum(dim=1)
#     num_positive_channels_ir, num_negative_channels_ir = positive_mask_ir.sum(dim=1), negative_mask_ir.sum(dim=1)

#     max_channels = max(rgb.size(1), ir.size(1))
#     all_channels_rgb, all_channels_ir = [], []

#     for batch_idx in range(rgb_sorted.size(0)):
#         # Get the number of positive and negative channels for this batch
#         num_pos_channels_rgb, num_neg_channels_rgb = num_positive_channels_rgb[batch_idx], num_negative_channels_rgb[batch_idx]
#         num_pos_channels_ir, num_neg_channels_ir = num_positive_channels_ir[batch_idx], num_negative_channels_ir[batch_idx]

#         # Slice positive and negative channels for this batch
#         positive_channels_rgb, negative_channels_rgb = rgb_sorted[batch_idx, :num_pos_channels_rgb], rgb_sorted[batch_idx, num_pos_channels_rgb:num_pos_channels_rgb + num_neg_channels_rgb]
#         positive_channels_ir, negative_channels_ir = ir_sorted[batch_idx, :num_pos_channels_ir], ir_sorted[batch_idx, num_pos_channels_ir:num_pos_channels_ir + num_neg_channels_ir]

#         if (positive_channels_rgb.size(0) > positive_channels_ir.size(0)) or (negative_channels_rgb.size(0) < negative_channels_ir.size(0)):
#             diff_pos = positive_channels_rgb.size(0) - positive_channels_ir.size(0)
#             diff_neg = negative_channels_ir.size(0) - negative_channels_rgb.size(0)
#             # print(diff_pos, diff_neg)
#             positive_channels_ir = torch.cat([positive_channels_ir, positive_channels_ir[:diff_pos]], dim=0)
#             negative_channels_ir = negative_channels_ir[diff_neg:]
#             # print(positive_channels_rgb.shape, negative_channels_rgb.shape, positive_channels_ir.shape, negative_channels_ir.shape)

#         elif (positive_channels_rgb.size(0) < positive_channels_ir.size(0)) or (negative_channels_rgb.size(0) > negative_channels_ir.size(0)):
#             diff_pos = positive_channels_ir.size(0) - positive_channels_rgb.size(0)
#             diff_neg = negative_channels_rgb.size(0) - negative_channels_ir.size(0)
#             # print(diff_pos, diff_neg)
#             positive_channels_rgb = torch.cat([positive_channels_rgb, positive_channels_rgb[:diff_pos]], dim=0)
#             negative_channels_rgb = negative_channels_rgb[diff_neg:]
#             # print(positive_channels_rgb.shape, negative_channels_rgb.shape, positive_channels_ir.shape, negative_channels_ir.shape)

#         # Ensure the channels are of the same length for both rgb and ir
#         while len(positive_channels_rgb) + len(negative_channels_rgb) < max_channels:
#             positive_channels_rgb = torch.cat([positive_channels_rgb, positive_channels_rgb[:1]], dim=0)

#         while len(positive_channels_ir) + len(negative_channels_ir) < max_channels:
#             positive_channels_ir = torch.cat([positive_channels_ir, positive_channels_ir[:1]], dim=0)

#         # positive_channels_rgb_w, positive_channels_ir_w = positive_channels_rgb * sa[batch_idx], positive_channels_ir * sa[batch_idx]
#         # negative_channels_rgb_w, negative_channels_ir_w = negative_channels_rgb * sa[batch_idx], negative_channels_ir * sa[batch_idx]
#         # positive_channels_rgb, positive_channels_ir = torch.add(positive_channels_rgb, positive_channels_rgb_w), torch.add(positive_channels_ir, positive_channels_ir_w)
#         # negative_channels_rgb, negative_channels_ir = torch.add(negative_channels_rgb, negative_channels_rgb_w), torch.add(negative_channels_ir, negative_channels_ir_w)
#         batch_channels_rgb, batch_channels_ir = torch.cat([positive_channels_rgb, negative_channels_rgb], dim=0), torch.cat([positive_channels_ir, negative_channels_ir], dim=0)

#         # assert batch_channels_rgb.size(0) != batch_channels_ir.size(0), "Mismatch in channels between RGB and IR"

#         all_channels_rgb.append(batch_channels_rgb)
#         all_channels_ir.append(batch_channels_ir)

#     # Concatenate positive and negative channels for each batch
#     all_positive_channels = torch.stack(all_channels_rgb, dim=0)
#     all_negative_channels = torch.stack(all_channels_ir, dim=0)

#     return all_positive_channels, all_negative_channels
    

def sort_and_split_channels(similarity, channels):
    sorted_values, sorted_indices = torch.sort(similarity, dim=1, descending=False)
    sorted_channels = torch.gather(channels, 1, sorted_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, channels.size(2), channels.size(3)))

    positive_mask = sorted_values > 0
    num_positive_channels = positive_mask.sum(dim=1)
    
    positive_channels = sorted_channels[:, :num_positive_channels.max()]
    negative_channels = sorted_channels[:, num_positive_channels.max():]

    return positive_channels, negative_channels


def equalize_channel_lengths_positive(rgb, ir):
    # 채널 수 확인
    rgb_channels = rgb.size(1) 
    ir_channels = ir.size(1)

    # 최대 채널 수에 맞추기 위한 추가 채널 수 계산
    extra_channels = abs(rgb_channels - ir_channels)
    min_channels = min(rgb_channels, ir_channels)
    if extra_channels > 0:
        rgb_extra, ir_extra = rgb[:,:extra_channels,:,:], ir[:,:extra_channels,:,:]
        if extra_channels <= min_channels:
            extra_fea = torch.max(rgb_extra[:, ::extra_channels, :, :], ir_extra[:, ::extra_channels, :, :])
            # RGB 텐서의 채널 수가 더 많은 경우
            if rgb_channels > ir_channels:
                ir = torch.cat([ir, extra_fea], dim=1)
            else:
                rgb = torch.cat([rgb, extra_fea], dim=1)
        
        else:
            extra_fea = torch.max(rgb_extra[:, ::rgb_channels, :, :], ir_extra[:, ::ir_channels, :, :])
            extra_fea = extra_fea[:,:extra_channels,:,:]
            if rgb_channels > ir_channels:
                ir = torch.cat([ir, extra_fea], dim=1)
            else:
                rgb = torch.cat([rgb, extra_fea], dim=1)

    return rgb, ir

def equalize_channel_lengths_negative(rgb, ir, extra_rgb, extra_ir):
    rgb, ir = rgb[:,:extra_rgb,:,:], ir[:,:extra_ir,:,:]
    return rgb, ir


def cosine_similarity_channel_sorting(rgb, ir, sa):
    sa = torch.sigmoid(sa)
    rsa = 1 - sa

    # Normalizing tensors
    sa_norm = F.normalize(sa, p=2, dim=(2, 3), eps=1e-12)
    rgb_norm = F.normalize(rgb, p=2, dim=(2, 3), eps=1e-12)
    ir_norm = F.normalize(ir, p=2, dim=(2, 3), eps=1e-12)

    # Compute cosine similarity
    similarity_rgb = (sa_norm * rgb_norm).sum(dim=(2, 3))
    similarity_ir = (sa_norm * ir_norm).sum(dim=(2, 3))

    positive_channels_rgb, negative_channels_rgb = sort_and_split_channels(similarity_rgb, rgb)
    positive_channels_ir, negative_channels_ir = sort_and_split_channels(similarity_ir, ir)

    max_channels = max(rgb.size(1), ir.size(1))
    positive_channels_rgb, positive_channels_ir = equalize_channel_lengths_positive(positive_channels_rgb, positive_channels_ir)
    extra_rgb, extra_ir = max_channels - positive_channels_rgb.size(1), max_channels - positive_channels_ir.size(1)
    negative_channels_rgb, negative_channels_ir = equalize_channel_lengths_negative(negative_channels_rgb, negative_channels_ir, extra_rgb, extra_ir)

    all_channels_rgb = torch.cat([positive_channels_rgb, negative_channels_rgb], dim=1)
    all_channels_ir = torch.cat([positive_channels_ir, negative_channels_ir], dim=1)

    return all_channels_rgb, all_channels_ir


# def sort_and_split_channels(similarity, channels):
#     sorted_values, sorted_indices = torch.sort(similarity, dim=1, descending=False)
#     sorted_channels = torch.gather(channels, 1, sorted_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, channels.size(2), channels.size(3)))

#     positive_mask = sorted_values > 0
#     num_positive_channels = positive_mask.sum(dim=1)
    
#     positive_channels = sorted_channels[:, :num_positive_channels.max()]
#     negative_channels = sorted_channels[:, num_positive_channels.max():]

#     return positive_channels, negative_channels

# def fill_conv1x1(in_channels, out_channels):
#     """Create a 1x1 convolution layer."""
#     return Conv(in_channels, out_channels, 1, 1)

# def adjust_channel_length_with_conv(channels, target_length):
#     current_length = channels.size(1)
#     if current_length < target_length:
#         # Define the 1x1 convolution
#         conv = fill_conv1x1(current_length, target_length - current_length)

#         # Apply the convolution
#         additional_channels = conv(channels)

#         # Concatenate the original and additional channels
#         channels = torch.cat([channels, additional_channels], dim=1)

#     return channels

# def cosine_similarity_channel_sorting(rgb, ir, sa):
#     sa = torch.sigmoid(sa)
#     rsa = 1 - sa

#     # Normalizing tensors
#     sa_norm = F.normalize(sa, p=2, dim=(2, 3), eps=1e-12)
#     rgb_norm = F.normalize(rgb, p=2, dim=(2, 3), eps=1e-12)
#     ir_norm = F.normalize(ir, p=2, dim=(2, 3), eps=1e-12)

#     # Compute cosine similarity
#     similarity_rgb = (sa_norm * rgb_norm).sum(dim=(2, 3))
#     similarity_ir = (sa_norm * ir_norm).sum(dim=(2, 3))

#     positive_channels_rgb, negative_channels_rgb = sort_and_split_channels(similarity_rgb, rgb)
#     positive_channels_ir, negative_channels_ir = sort_and_split_channels(similarity_ir, ir)

#     max_channels = max(rgb.size(1), ir.size(1))

#     # Adjust channel lengths using 1x1 convolution
#     positive_channels_rgb = adjust_channel_length_with_conv(positive_channels_rgb, max_channels)
#     negative_channels_rgb = adjust_channel_length_with_conv(negative_channels_rgb, max_channels)
#     positive_channels_ir = adjust_channel_length_with_conv(positive_channels_ir, max_channels)
#     negative_channels_ir = adjust_channel_length_with_conv(negative_channels_ir, max_channels)

#     all_channels_rgb = torch.cat([positive_channels_rgb, negative_channels_rgb], dim=1)
#     all_channels_ir = torch.cat([positive_channels_ir, negative_channels_ir], dim=1)

#     return all_channels_rgb, all_channels_ir


# def optimized_cosine_similarity_channel_sorting(rgb, ir, sa):
#     rsa = 1 - sa
    
#     sa_norm = F.normalize(sa, p=2, dim=(2,3), eps=1e-12)
#     rgb_norm = F.normalize(rgb, p=2, dim=(2,3), eps=1e-12)
#     ir_norm = F.normalize(ir, p=2, dim=(2,3), eps=1e-12)

#     similarity_rgb = (sa_norm * rgb_norm).sum(dim=(2,3))
#     similarity_ir = (sa_norm * ir_norm).sum(dim=(2,3))

#     _, sorted_indices_rgb = torch.sort(similarity_rgb, dim=1, descending=True)
#     _, sorted_indices_ir = torch.sort(similarity_ir, dim=1, descending=True)

#     rgb_sorted = torch.gather(rgb, 1, sorted_indices_rgb.unsqueeze(2).unsqueeze(3).expand(-1, -1, rgb.size(2), rgb.size(3)))
#     ir_sorted = torch.gather(ir, 1, sorted_indices_ir.unsqueeze(2).unsqueeze(3).expand(-1, -1, ir.size(2), ir.size(3)))

#     max_channels = max(rgb.size(1), ir.size(1))
#     all_channels_rgb, all_channels_ir = [], []

#     for batch_idx in range(rgb_sorted.size(0)):
#         rgb_channels = torch.cat([rgb_sorted[batch_idx] * sa[batch_idx], rgb_sorted[batch_idx] * rsa[batch_idx]], dim=0)
#         ir_channels = torch.cat([ir_sorted[batch_idx] * sa[batch_idx], ir_sorted[batch_idx] * rsa[batch_idx]], dim=0)

#         # Pad channels to ensure they all have the same size
#         if rgb_channels.size(0) < max_channels:
#             padding_size = max_channels - rgb_channels.size(0)
#             padding = torch.zeros((padding_size, rgb_channels.size(1), rgb_channels.size(2)), device=rgb_channels.device)
#             rgb_channels = torch.cat([rgb_channels, padding], dim=0)

#         if ir_channels.size(0) < max_channels:
#             padding_size = max_channels - ir_channels.size(0)
#             padding = torch.zeros((padding_size, ir_channels.size(1), ir_channels.size(2)), device=ir_channels.device)
#             ir_channels = torch.cat([ir_channels, padding], dim=0)

#         all_channels_rgb.append(rgb_channels)
#         all_channels_ir.append(ir_channels)

#     return torch.stack(all_channels_rgb, dim=0), torch.stack(all_channels_ir, dim=0)


class CSFM(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        self.spatial_attention = SpatialAttentionModule()
    
    def forward(self, x):
        rgb, ir = x[0], x[1]
        rgb_sa, ir_sa = self.spatial_attention(rgb), self.spatial_attention(ir)
        sa = torch.cat([rgb_sa, ir_sa], 1)
        sa = torch.max(sa[:, ::2 , : , : ], sa[: , 1::2, : ,: ])
        rgb_cs, ir_cs = cosine_similarity_channel_sorting(rgb, ir, sa)
        output = torch.add(rgb_cs, ir_cs)
        return output
    

class CSCR(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        rgb, ir = x[0], x[1]
        rgb_cap, rgb_cmp = torch.mean(rgb, dim=1, keepdim=True), torch.max(rgb, dim=1, keepdim=True)[0]
        ir_cap, ir_cmp = torch.mean(ir, dim=1, keepdim=True), torch.max(ir, dim=1, keepdim=True)[0]
        x1_cp, x2_cp = torch.cat([rgb_cap, rgb_cmp], dim=1), torch.cat([ir_cap, ir_cmp], dim=1)
        cp = torch.add(x1_cp, x2_cp)
        sa = torch.max(cp[:, ::2 , : , : ], cp[: , 1::2, : ,: ])
        rgb_cs, ir_cs = cosine_similarity_channel_sorting(rgb, ir, sa)
        sa_sig = self.sigmoid(sa)
        rgb_cs, ir_cs = rgb_cs * sa_sig, ir_cs * sa_sig
        # output = torch.add(rgb_cs, ir_cs)
        return rgb_cs, ir_cs



class IAFA(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        rgb, ir = x[0], x[1]
        f_d = rgb - ir
        d_gap, d_gmp = self.sigmoid(self.gap(f_d)), self.sigmoid(F.adaptive_max_pool2d(f_d, (1,1)))
        d_gp = torch.add(d_gap, d_gmp) / 2
        # rgb_gap, rgb_gmp = self.sigmoid(self.gap(rgb)), self.sigmoid(F.adaptive_max_pool2d(rgb))
        # ir_gap, ir_gmp = self.sigmoid(self.gap(ir)), self.sigmoid(F.adaptive_max_pool2d(ir))

        rgb_dmaf, ir_dmaf = rgb * d_gp, ir * d_gp
        # rgb, ir = torch.add(rgb, ir_dmaf), torch.add(ir, rgb_dmaf)
        return rgb_dmaf, ir_dmaf
    

# class TransformerModule(nn.Module):
#     def __init__(self, in_channels, num_heads=4, num_layers=2):
#         super(TransformerModule, self).__init__()
#         self.in_channels = in_channels
#         self.num_heads = num_heads
#         self.num_layers = num_layers

#         # Transformer encoder layers
#         encoder_layers = nn.TransformerEncoderLayer(d_model=self.in_channels, nhead=num_heads)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

#     def forward(self, x):
#         # x: (batch_size, in_channels, 1, 1)
#         x = x.flatten(2)  # (batch_size, in_channels, 1)
#         x = x.permute(2, 0, 1)  # (1, batch_size, in_channels) for transformer
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 2, 0).view(-1, self.in_channels, 1, 1)  # Reshape back
#         return x


class SelfAttention_jb(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out


class CIC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.transformer_module = SelfAttention_jb(in_channels * 2) 
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        rgb, ir = x[0], x[1]
        f_d = rgb - ir
        d_gap = self.sigmoid(self.gap(f_d))
        r_gap, i_gap = self.gap(rgb), self.gap(ir)

        combined_feature = torch.cat([r_gap, i_gap], dim=1)
        transformer_output = self.transformer_module(combined_feature)
        r_gap_t, i_gap_t = torch.split(transformer_output, self.in_channels, dim=1)

        rgb, ir = torch.add(rgb, ir * d_gap), torch.add(ir, rgb * d_gap)
        rgb_output, ir_output = torch.add(rgb, rgb * r_gap_t), torch.add(ir, ir * i_gap_t)
        # w_r, w_i = rgb * d_gap, ir * d_gap
        # w_r, w_i = rgb * i_gap_t, ir * r_gap_t
        # rgb_dmaf, ir_dmaf = torch.add(rgb, w_r), torch.add(ir, w_i)
        return rgb_output, ir_output
    

class MADL(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.transformer_module = SelfAttention_jb(in_channels * 2) 
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        rgb, ir = x[0], x[1]
        f_d = rgb - ir
        d_gap = self.sigmoid(self.gap(f_d))
        r_gap, i_gap = self.gap(rgb), self.gap(ir)

        combined_feature = torch.cat([r_gap, i_gap], dim=1)
        transformer_output = self.transformer_module(combined_feature)
        r_gap_t, i_gap_t = torch.split(transformer_output, self.in_channels, dim=1)

        rgb, ir = torch.add(rgb, ir * d_gap), torch.add(ir, rgb * d_gap)
        # rgb_output, ir_output = torch.add(rgb, rgb * r_gap_t), torch.add(ir, ir * i_gap_t)
        # w_r, w_i = rgb * i_gap_t, ir * r_gap_t
        # rgb_dmaf, ir_dmaf = torch.add(rgb, w_r), torch.add(ir, w_i)
        return rgb, ir
    

class MACSA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.transformer_module = SelfAttention_jb(in_channels * 2) 
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        rgb, ir = x[0], x[1]
        f_d = rgb - ir
        d_gap = self.sigmoid(self.gap(f_d))
        r_gap, i_gap = self.gap(rgb), self.gap(ir)

        combined_feature = torch.cat([r_gap, i_gap], dim=1)
        transformer_output = self.transformer_module(combined_feature)
        r_gap_t, i_gap_t = torch.split(transformer_output, self.in_channels, dim=1)

        # rgb, ir = torch.add(rgb, ir * d_gap), torch.add(ir, rgb * d_gap)
        rgb_output, ir_output = torch.add(rgb, rgb * r_gap_t), torch.add(ir, ir * i_gap_t)
        # rgb_dmaf, ir_dmaf = torch.add(rgb, w_r), torch.add(ir, w_i)
        return rgb_output, ir_output
    
