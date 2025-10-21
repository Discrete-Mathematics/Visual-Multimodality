# FTransUNet在Vaihingen数据集上的复现结果
| 项目        | 分析                                       |
| --------- | ---------------------------------------- |
| PA      | 92.22% |
| mF1 | 0.9072|
| mIoU  | 0.8343|
| Kappa     | 0.895|
# FTransUNet在MUSeg数据集上的实验数据
## 实验一
一开始我在数据预处理时只保留了数据集中的图片的中间区域，这导致训练效果非常不理想，测试cpa也指示出在测试时大部分类别都没有被成功预测到。  

训练/测试配置：
``` python
N_CLASSES = 16
EPOCHS = 50
BATCH_SIZE = 12
Stride_Size = 128
WINDOW_SIZE = (256, 256)

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = N_CLASSES
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))

net = ViT_seg(config_vit, img_size=256, num_classes=16).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))

base_lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
scaler = amp.GradScaler()

# 创建模型并加载预训练权重
net = ViT_seg(config_vit, img_size=256, num_classes=16).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))
```
|指标|数据|
|---------|-----|
|训练集上的最佳PA|0.57|
|最终测试PA|0.45|
|最终测试mPA |0.07|
|最终测试mIoU|0.03|
## 实验二
后来我选择从左上角开始，每次向右/下移动256个像素来裁剪，使图片左上部分的区域得到了充分利用，但训练效果仍不是很理想，测试cpa仍指示出在测试时有好几个类别没有被成功预测到，可能是因为右下部分没有很好利用。  

训练/测试配置：
```python
N_CLASSES = 16
EPOCHS = 50
BATCH_SIZE = 12
Stride_Size = 128
WINDOW_SIZE = (256, 256)

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = N_CLASSES
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))

net = ViT_seg(config_vit, img_size=256, num_classes=16).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))

base_lr = 0.001
optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005, amsgrad=False)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40], gamma=0.1)
scaler = amp.GradScaler()

# 创建模型并加载预训练权重
net = ViT_seg(config_vit, img_size=256, num_classes=16).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))
```
|指标|数据|
|---------|-----|
|训练集上的最佳PA|0.71|
|最终测试PA|0.63|
|最终测试mPA |0.18|
|最终测试mIoU|0.12|
## 实验三
吸取实验二的教训，我打算使用albumentations库来对数据集做裁剪和增强，但训练效果反而更差了，测试cpa指示出只有两个类别被成功预测了。

训练/测试配置：
```python
N_CLASSES = 16
EPOCHS = 205
BATCH_SIZE = 32
Stride_Size = 128
WINDOW_SIZE = (256, 256)

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = N_CLASSES
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))

net = ViT_seg(config_vit, img_size=256, num_classes=16).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))

base_lr = 0.01
optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.8, 0.999), eps=1e-8, weight_decay=0.0005, amsgrad=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150], gamma=0.5)
scaler = amp.GradScaler()

# 创建模型并加载预训练权重
net = ViT_seg(config_vit, img_size=256, num_classes=16).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))
```
|指标|数据|
|---------|-----|
|训练集上的最佳PA|0.5555|
|最终测试PA|0.5555|
|最终测试mPA |0.0985|
|最终测试mIoU|0.0582|
## 实验四
我在复盘实验二时发现，由于2048和1536都能被256整除，因此实验二中的图片右下部分的区域应该也得到了利用，因此我对实验二的代码做了如下修改：
1. 改进数据集类：支持多尺度训练、添加上下文扩展、随机裁剪
2. 改进训练函数：使用多尺度训练和上下文扩展
3. 改进测试函数：使用高斯加权融合和上下文扩展
4. 引入了梯度裁剪使训练过程更稳健
5. 使用更安全的交叉熵损失函数

训练/测试配置：
```python
N_CLASSES = 16
BATCH_SIZE = 32
EPOCHS = 205 #实际上只训练了40轮
WINDOW_SIZE = (256, 256)
OVERLAP_RATIO = 0.5  # 50%重叠
STRIDE_SIZE = int(WINDOW_SIZE[0] * (1 - OVERLAP_RATIO))  # 128
CONTEXT_MARGIN = 64  # 上下文扩展区域

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = N_CLASSES
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))

net = ViT_seg(config_vit, img_size=256, num_classes=16).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))

base_lr = 0.0001 * 1.414
optimizer = optim.AdamW(
    net.parameters(), 
    lr=base_lr, 
    weight_decay=0.01, 
    betas=(0.9, 0.999),
    eps=1e-7,  # 增加epsilon值提高稳定性
    amsgrad=True
)

# 使用更保守的余弦退火学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=50, 
    eta_min=base_lr/1000  # 更小的最小学习率
)

# 创建混合精度训练的scaler（更保守的设置）
scaler = amp.GradScaler(
    init_scale=65536.0,  # 初始缩放因子
    growth_factor=1.1,    # 更保守的增长因子
    backoff_factor=0.5,   # 更积极的回退因子
    growth_interval=1000  # 更长的增长间隔
)

# 创建模型并加载预训练权重
net = ViT_seg(config_vit, img_size=256, num_classes=16).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))

torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) #梯度裁剪
```
这次训练效果还行，虽然只训练了40轮，但训练集上的PA已稳定在了90%以上，在测试时也没有表现出明显的过拟合（测试cpa指示出只有两个类别没有预测出来）；最大的问题是训练得太慢了，训练40轮都已经花了将近2天时间。和彭煜交流后确定原因为：由于我将2048x1536的图片裁剪为了8x7个256x256的图片，这导致训练需要的时间开销变为了56倍。因此我打算改进数据预处理方式后再做一次实验。
|指标|数据|
|---------|-----|
|40轮中训练集上的最佳mIoU|0.6532|
|最终测试PA|0.9234|
|最终测试mPA |0.7287|
|最终测试mIoU|0.6572|

## 实验五
我在前面四次实验基础上改进的地方如下：
1. 预处理时按50%概率水平翻转RGB图像
2. 预处理时从预设比例数组[0.5, 0.75, 1.0, 1.25, 1.5]中随机选择缩放比例
3. 预处理时等比例缩放图像到中间尺寸，边缘用0填充
（故输入的是原始图像而不是裁剪后的图像）
4. 预处理时使用`(input - mean)/std`归一化
5. 使用数据集划分的训练集和测试集来做训练和测试
训练/测试配置：
```python
N_CLASSES = 16
BATCH_SIZE = 32
EPOCHS = 500 #实际上只训练了46轮
OVERLAP_RATIO = 0.75  # 75%重叠
WINDOW_SIZE = (256, 256)
STRIDE_SIZE = int(WINDOW_SIZE[0] * (1 - OVERLAP_RATIO))  # 128
CONTEXT_MARGIN = 64  # 上下文扩展区域

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = N_CLASSES
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))

# 创建模型并加载预训练权重
net = ViT_seg(config_vit, img_size=256, num_classes=16).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path, allow_pickle=True))

# 优化器配置 - 使用AdamW + AMSGrad
base_lr = 0.000005
optimizer = optim.AdamW(
    net.parameters(), 
    lr=base_lr, 
    weight_decay=0.01, 
    betas=(0.9, 0.999),
    eps=1e-6,  # 增加epsilon值提高稳定性
    amsgrad=False
)

# 使用更保守的余弦退火学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=50, 
    eta_min=base_lr/1000  # 更小的最小学习率
)

# 创建混合精度训练的scaler（更保守的设置）
scaler = amp.GradScaler(
    init_scale=4096.0,  # 初始缩放因子
    growth_factor=1.1,    # 更保守的增长因子
    backoff_factor=0.5,   # 更积极的回退因子
    growth_interval=1000  # 更长的增长间隔
)

# 添加梯度裁剪（重要安全措施）
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
```
这次训练的特点有：每轮epoch训练的速度很快，但测试的速度很慢，loss收敛得也非常慢（46epochs时收敛得还没实验四好）。总体而言，训练效果不好：
|指标|数据|
|---------|-----|
|最终测试PA|0.5015|
|最终测试mPA |0.0969|
|最终测试mIoU|0.0573|                            
|最终测试cpa|[9.35987050e-01,0.00000000e+00,1.26462969e-01,5.29001403e-05,1.40515773e-07,5.98697837e-03,1.88051816e-01,1.91142612e-02,7.59413279e-07,1.16265302e-02,0.00000000e+00,7.09955939e-03,0.00000000e+00,2.92727407e-05,4.78121542e-02,2.08241912e-01]|
## 实验六
重做实验四，这次严格按照数据集论文划分训练集和测试集，其他的都不做修改。  
共训练了40左右个epoch，训练到第22个epoch时测试集上mIoU的提升就不再明显了。
|指标|数据|
|---------|-----|
|最终测试PA|0.7186|
|最终测试mPA |0.3962|
|最终测试mIoU|0.2849|                            
|最终测试cpa|[8.70496639e-01,
0.00000000e+00,
6.54237581e-01,
6.08614415e-01,
7.81270652e-01,
4.25294674e-01,
1.21331148e-06,
5.93255746e-01,
5.93897382e-01,
5.65348743e-01,
0.00000000e+00,
8.98456906e-05,
1.24307636e-06,
0.00000000e+00,
4.62486630e-01,
7.84024260e-01]|

其中，检测效果最差的几个类别是person, door, container。
## 实验七
考虑到先前实验结果中存在的分割效果都不佳，然后存在小物体、杂乱地方、暗物体检测困难等问题，我打算通过moe来做图像增强，具体内容是将数据集中所有图片增强（肉眼看上去的效果是变亮了），然后再使用增强后的图片做训练、验证、测试，相当于做成了moe+FTransUNet的级联模型，代码使用的还是实验六的代码，并训练了完整的205轮（模型在189轮时在验证集上的mIoU就不再增加），效果如下：

|指标|数据|
|-|-|
|最终测试pa|0.7552|                                                                
|最终测试cpa|[0.91433111 0.35677871 0.70230885 0.67153295 0.76406359 0.58117805 0.38943826 0.49799002 0.62039139 0.54094898 0.25825255 0.25773983 0.4702679  0.16346112 0.47233386 0.7841275]|
|最终测试mpa|0.5278|
|最终测试mIoU|0.3926|

整体而言，可能因为训练轮数的增加，测试的cpa比起先前的实验有所提升，但person, door, tools and materials, container等类的cpa较低。我觉得可以看出person, door,container是较难检测的类别了，这也可能与我预处理时对图片做的裁剪有关。

除此之外，分割结果可视化表明分割结果仍然不理想：类别检测错误十分严重，也存在轮廓不清晰、区域错误等问题。
## 实验八
这次尝试了把moe+FTransUNet做成端到端模型，不过训练起来有些困难，loss总是NaN，接下来我打算等期末考完了就来调参。