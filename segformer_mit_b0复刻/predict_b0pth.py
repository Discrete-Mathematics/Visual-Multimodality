'''目标：实现通过加载.pth文件，对输入图像进行语义分割，并将结果可视化显示'''
'''复刻的文件有predict.py,segformer.py,backbone.py'''
'''只有mode=predict功能，检测单张图'''
import torch
from PIL import Image
from torch import nn as nn
import torch.nn.functional as F
import math
import warnings
from functools import partial
import cv2

'''mlp:(B, C, H, W)->(B, H*W, C),2048->768
convmodule:逐点卷积，归一，relu
segformerhead:(B, C, H, W)->(B, H*W, C)->(B, C, H, W),c1到c4上采样到c1的大小，在通道上拼接，最后通过convmodule融合特征图，最后通过1x1卷积预测类别
segformer:将输入图像通过backbone提取特征图，segformerhead进行解码，回原图大小'''
class MLP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=768):
        super(MLP, self).__init__()
        self.proj=nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x=x.flatten(2).transpose(1,2)
        x=self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv=nn.Conv2d(c1,c2, k, s, p, groups=g, bias=False)
        self.bn=nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act=nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class SegFormerHead(nn.Module):
    def __init__(self,num_classes=20,in_channels=[32,64,160,256],embedding_dim=768,dropout_ratio=0.1):
        super(SegFormerHead,self).__init__()
        c1,c2,c3,c4=in_channels

        self.linear_c4=MLP(in_channels=c4, out_channels=embedding_dim)
        self.linear_c3=MLP(in_channels=c3, out_channels=embedding_dim)  
        self.linear_c2=MLP(in_channels=c2, out_channels=embedding_dim)
        self.linear_c1=MLP(in_channels=c1, out_channels=embedding_dim)

        self.linear_fuse=ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )
        self.linear_pred=nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout=nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1,c2,c3,c4=inputs
        _c4=self.linear_c4(c4).permute(0,2,1).reshape(c4.shape[0],-1,c4.shape[2],c4.shape[3])
        _c4=F.interpolate(_c4,size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c3=self.linear_c3(c3).permute(0,2,1).reshape(c3.shape[0],-1,c3.shape[2],c3.shape[3])
        _c3=F.interpolate(_c3,size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c2=self.linear_c2(c2).permute(0,2,1).reshape(c2.shape[0],-1,c2.shape[2],c2.shape[3])
        _c2=F.interpolate(_c2,size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c1=self.linear_c1(c1).permute(0,2,1).reshape(c1.shape[0],-1,c1.shape[2],c1.shape[3])

        _c=self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x=self.dropout(_c)
        x=self.linear_pred(x)
        return x
    
class Segformer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False):
        super(Segformer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256]
        }[phi]
        self.backbone   = {
            'b0': mit_b0
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256
        }[phi]
        self.decoder = SegFormerHead(num_classes=num_classes, in_channels=self.in_channels, embedding_dim=self.embedding_dim)

    def forward(self, inputs):
        h,w= inputs.shape[2], inputs.shape[3]
        x= self.backbone(inputs)
        x= self.decoder(x)
        x= F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

'''dwconv:分层卷积
逆变换采样：mean=0., std=1., a=-2., b=2.
堆叠图像嵌入：卷积为四分之一的图像大小，变换形状，layernorm在通道上归一化
mlp_:linear，dwconv，激活函数gelu，dropout,linear,dropout
attention:qkv线性变换，图像进行下采样
block:归一化，注意力机制，归一化，mlp，drop_path
drop_path:张量以drop_prob概率置零''' 
def InverseTransform_Sampling(tensor,mean=0.,std=1.,a=-2.,b=2.):
    def norm_cdf(x):
        return 0.5*(1+torch.erf((x-mean)/(std*torch.sqrt(torch.tensor(2.0)))))
    if (mean<a-2*std) or (mean>b+2*std):
        warnings.warn("mean is out of range, please check the parameters",stacklevel=2)
    with torch.no_grad():
        l=norm_cdf((a-mean)/std)    #小于 a 的概率
        r=norm_cdf((b-mean)/std)
        tensor.uniform_(2*l-1, 2*r-1)   #在范围 [2*l-1, 2*u-1] 内生成均匀分布的随机数
        tensor.erfinv_()
        tensor.mul_(std*math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(a, b)
    return tensor
    
class Gelu(nn.Module):
    def __init__(self):
        super(Gelu, self).__init__()
    def forward(self, x):
        return 0.5*x*(1+torch.tanh(math.sqrt(2/math.pi)*(x+0.044715*torch.pow(x,3))))

def _init_weights(m):
    if isinstance(m,nn.Linear):
        InverseTransform_Sampling(m.weight,std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m,nn.Conv2d):
        fan_out=m.kernel_size[0]*m.kernel_size[1]*m.out_channels    #kaiming正态分布初始化
        fan_out //= m.groups
        m.weight.data.normal_(0,math.sqrt(2.0/fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    if isinstance(m,nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, patch_size=7, stride=4):
        super(OverlapPatchEmbed, self).__init__()
        patch_size=(patch_size,patch_size)
        self.proj=nn.Conv2d(in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,
                            stride=stride,padding=(patch_size[0]//2,patch_size[1]//2))
        self.norm=nn.LayerNorm(embed_dim)   #用于transformer的归一化
        self.apply(_init_weights)

    def forward(self, x):
        x=self.proj(x)
        b,c,h,w=x.shape
        x=x.flatten(2).transpose(1,2)
        x=self.norm(x)
        return x,h,w
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,sr_ratio=1):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim=dim
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale=qk_scale or head_dim**-0.5
        self.q=nn.Linear(dim,dim,bias=qkv_bias)
        self.kv=nn.Linear(dim, dim*2, bias=qkv_bias)    #减小内存
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim, dim)
        self.proj_drop=nn.Dropout(proj_drop)

        self.sr_ratio=sr_ratio  #空间分辨率比率为1，表示不进行下采样
        if sr_ratio>1:  #图像大小为(H,W)，下采样后大小为(H//sr_ratio,W//sr_ratio)
            self.sr=nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm=nn.LayerNorm(dim)

        self.apply(_init_weights)

    def forward(self,x,h,w):
        b,n,c=x.shape
        q=self.q(x).reshape(b,n,self.num_heads,c//self.num_heads).permute(0,2,1,3)  #q:(B,num_heads,N,c//num_heads)

        if self.sr_ratio>1:
            x_=x.permute(0,2,1).reshape(b,c,h,w)
            x_=self.sr(x_).reshape(b,c,-1).permute(0,2,1)   #x_:(B,H//sr_ratio*W//sr_ratio,c)
            x_=self.norm(x_)
            kv=self.kv(x_).reshape(b,-1,2,self.num_heads,c//self.num_heads).permute(2,0,3,1,4)
        else:   
            kv=self.kv(x).reshape(b,-1,2,self.num_heads,c//self.num_heads).permute(2,0,3,1,4)
        k,v=kv[0],kv[1]   #k,v:(B,num_heads,N,c//num_heads)

        attn=q @ k.transpose(-2,-1) * self.scale   #attn:(B,num_heads,N,N)
        attn=attn.softmax(dim=-1)
        attn=self.attn_drop(attn)

        x=attn @ v   #x:(B,num_heads,N,c//num_heads)
        x=x.transpose(1,2).reshape(b,n,c)   #x:(B,N,c)
        x=self.proj(x) 
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=Gelu, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(Block, self).__init__()
        self.norm1=norm_layer(dim)
        self.attn=Attention(dim,num_heads,qkv_bias,qk_scale,attn_drop,drop,sr_ratio)
        self.norm2=norm_layer(dim)
        self.mlp=MLP_(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(_init_weights)

    def forward(self, x, h, w):
        x=x+self.drop_path(self.attn(self.norm1(x), h, w))  #注意力机制
        x=x+self.drop_path(self.mlp(self.norm2(x), h, w))  #多层感知机
        return x

class DwConv(nn.Module):
    def __init__(self, dim=768):
        super(DwConv, self).__init__()
        self.dwconv=nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

    def forward(self,x,h,w):
        b,n,c=x.shape
        x=x.permute(0,2,1).reshape(b,c,h,w)
        self.dwconv(x)
        x=x.flatten(2).transpose(1,2)
        return x

class MLP_(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=Gelu, drop=0.):
        super(MLP_, self).__init__()
        out_features=out_features or in_features
        hidden_features=hidden_features or in_features
        self.fc1=nn.Linear(in_features, hidden_features)
        self.dwconv=DwConv(hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features, out_features)
        self.drop=nn.Dropout(drop)
        self.apply(_init_weights)
    
    def forward(self, x, h, w):
        x=self.fc1(x)
        x=self.dwconv(x, h, w)  #dwconv操作
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x
    
class DropPath(nn.Module):
    def __init__(self,drop_prob=0.,scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob=drop_prob
        self.scale_by_keep=scale_by_keep
        
    def forward(self, x,training=False):
        if self.drop_prob==0. or not self.training:
            return x
        keep_prob=1-self.drop_prob
        shape=(x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor=x.new_empty(shape).bernoulli_(keep_prob)  
        if self.scale_by_keep and keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

'''MixVisionTransformer:
                        512, 512, 3 => 128, 128, 32 => 16384, 32    4*
                        128, 128, 32 => 64, 64, 64 => 4096, 64      2*
                        64, 64, 64 => 32, 32, 160 => 1024, 160      2*
                        32, 32, 160 => 16, 16, 256 => 256, 256      2*
                        总共下采样32倍
    mit_b0配置:embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4]'''   
class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super(MixVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.depths = depths
        dpr= [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] #得到浅层到深层的特征信息

        self.patch_embed1=OverlapPatchEmbed(in_channels=in_chans,embed_dim=embed_dims[0],patch_size=7,stride=4)
        self.patch_embed2=OverlapPatchEmbed(in_channels=embed_dims[0],embed_dim=embed_dims[1],patch_size=3,stride=2)
        self.patch_embed3=OverlapPatchEmbed(in_channels=embed_dims[1],embed_dim=embed_dims[2],patch_size=3,stride=2)
        self.patch_embed4=OverlapPatchEmbed(in_channels=embed_dims[2],embed_dim=embed_dims[3],patch_size=3,stride=2)
        
        cur=0
        self.block1=nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],num_heads=num_heads[0],mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop_rate,attn_drop=attn_drop_rate,
                    drop_path=dpr[cur+i],norm_layer=norm_layer,sr_ratio=sr_ratios[0]
                )for i in range(depths[0])
            ]
        )
        cur+=depths[0]
        self.block2=nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],num_heads=num_heads[1],mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop_rate,attn_drop=attn_drop_rate,
                    drop_path=dpr[cur+i],norm_layer=norm_layer,sr_ratio=sr_ratios[1]
                )for i in range(depths[1])
            ]
        )
        cur+=depths[1]
        self.block3=nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],num_heads=num_heads[2],mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop_rate,attn_drop=attn_drop_rate,
                    drop_path=dpr[cur+i],norm_layer=norm_layer,sr_ratio=sr_ratios[2]
                )for i in range(depths[2])
            ]
        )
        cur+=depths[2]
        self.block4=nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],num_heads=num_heads[3],mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop_rate,attn_drop=attn_drop_rate,
                    drop_path=dpr[cur+i],norm_layer=norm_layer,sr_ratio=sr_ratios[3]
                )for i in range(depths[3])
            ]
        )

        self.norm1=norm_layer(embed_dims[0])
        self.norm2=norm_layer(embed_dims[1])
        self.norm3=norm_layer(embed_dims[2])
        self.norm4=norm_layer(embed_dims[3])

        self.apply(_init_weights)

    def forward(self, x):
        b = x.shape[0]
        outs = []

        # stage 1
        x, h, w = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h, w)
        x1 = self.norm1(x)
        x1_out = x1.permute(0, 2, 1).reshape(b, -1, h, w).contiguous()
        outs.append(x1_out)

        # stage 2
        x, h, w = self.patch_embed2(x1_out)
        for blk in self.block2:
            x = blk(x, h, w)
        x2 = self.norm2(x)
        x2_out = x2.permute(0, 2, 1).reshape(b, -1, h, w).contiguous()
        outs.append(x2_out)

        # stage 3
        x, h, w = self.patch_embed3(x2_out)
        for blk in self.block3:
            x = blk(x, h, w)
        x3 = self.norm3(x)
        x3_out = x3.permute(0, 2, 1).reshape(b, -1, h, w).contiguous()
        outs.append(x3_out)

        # stage 4
        x, h, w = self.patch_embed4(x3_out)
        for blk in self.block4:
            x = blk(x, h, w)
        x4 = self.norm4(x)
        x4_out = x4.permute(0, 2, 1).reshape(b, -1, h, w).contiguous()
        outs.append(x4_out)

        return outs
    
class mit_b0(MixVisionTransformer):
    def __init__(self, pretrained = False):
        super(mit_b0, self).__init__(
            embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load(r"D:\fuwuqi\segformer集合\segformer_mit_b0复刻\segformer_b0_backbone_weights.pth"), strict=False)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
def preprocess_input(image):
    image -= np.array([123.675, 116.28, 103.53], np.float32)
    image /= np.array([58.395, 57.12, 57.375], np.float32)
    return image

class SegFormer_Segmentation(object):
    _defaults = {
        "model_path": r"D:\fuwuqi\segformer集合\segformer_mit_b0复刻\segformer_b0_weights_voc.pth",  # 修改为你的模型路径
        "num_classes": 21,
        "phi": "b0",
        "input_shape": [512, 512],
        "mix_type": 0,
        "cuda": False,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        # 固定使用21类的颜色表（PASCAL VOC）
        self.colors = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), 
            (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), 
            (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), 
            (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)
        ]
        
        # 初始化模型
        self.net = Segformer(num_classes=self.num_classes, phi=self.phi, pretrained=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        # 替换decode_head为decoder
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("decode_head."):
                new_k = "decoder." + k[len("decode_head."):]
            else:
                new_k = k
            new_state_dict[new_k] = v
        self.net.load_state_dict(new_state_dict, strict=False)
        self.net = self.net.eval()

    def detect_image(self, image, count=False, name_classes=None):
        # 图像预处理
        image = cvtColor(image)
        old_img = image.copy()
        h, w = image.size[1], image.size[0]
        
        # 调整尺寸并归一化
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        # 模型推理
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2 ): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2 ): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (w, h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        # 结果可视化（默认mix_type=0，与原图混合）
        seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [h, w, -1])
        result = Image.fromarray(np.uint8(seg_img))
        return Image.blend(old_img, result, 0.7)

if __name__ == "__main__":
    segformer = SegFormer_Segmentation()
    num_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
                  "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
                  "horse", "motorbike", "person", "pottedplant", "sheep", 
                  "sofa", "train", "tvmonitor"]
    
    while True:
        img_path = input("请输入图像路径（输入q退出）：")
        if img_path.lower() == 'q':
            break
            
        try:
            image = Image.open(img_path)
            result = segformer.detect_image(image)
            result.show()
        except Exception as e:
            print("图像路径错误或无法打开，请重试", "错误信息：", e)
