# 简易的Vision Transformer 图像分类

基于PyTorch实现最基础的Vision Transformer (ViT) 网络结构，并用于花卉图片分类任务。


## 数据集

下载 [flower_photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) 并解压到 `datasets/flower_photos` 目录下。

## 训练模型

运行：

```sh
python train.py
```
main()函数内可修改一部分参数

## 模型结构

模型结构见 [`vit.py`](vit.py)，为最基础的 vit 网络结构
