# SelectiveMine 冻结源码包（2026-10-23）

这是当前研究候选的**源码复现包**，不是已验证优于基线的最终模型发布。包内不含数据图像、ImageNet权重或正式研究权重，也不包括整套历史队列和CMX。论文方法底稿见 `mine_selective/METHOD.md`。所有代码执行、模型检查和训练仅在获准的AutoDL上进行。

## 内容及环境

- `mine_relay/`、`mine_selective/`：与当前训练逐文件一致的26份Python源码，包含数据、模型、训练、完成核验、评估及地点配对工具。
- `tools/selective_inference_bundle_20261023.py`：经云端软件模型检查的推理权重导出/加载工具；正式模型仍需完成后核验。
- `splits/`：原始开发训练/验证名单，1281/314图，保留字节与哈希。
- `records/preregistration_20261023.json`：当时的实验登记，仅为来源记录；里面的绝对路径和进程辅助文件不表示本包包含整套队列。
- `manifest.json`：本包文件哈希、模型来源、初始化权重哈希和观察到的环境。
- `recipes.json`：两组主实验的完整参数及路径变量，不会自动启动训练。

本次实际环境为Python3.12、PyTorch2.8.0+cu128、NumPy2.3.2、Pillow11.3.0。保持所租镜像的PyTorch，不用本包替换CUDA栈。核心模型不依赖mmcv/mmsegmentation/timm或额外自定义CUDA扩展；绘图工具未包含在这个核心包中。

## 重新运行实验

先核对manifest中的文件、数据划分和ImageNet权重哈希。数据目录需有原始RGB/Depth/Label及官方train.txt、val.txt；原始深度保留16位。将recipes中的数据、输出、预算和ImageNet路径变量替换为实际路径，在本包目录执行 `python -m mine_selective.train` 后接对应参数。`fusion=logits`必须显式保留，不能使用训练入口默认的relay。

预算记录必须对应用户实际授权；本包不创建预算、不复制新的开始时间，也不授权重跑或新增GPU费用。若从本轮结束后的其他窗口重跑，需要新的真实授权记录。保持同一120轮日程、增强、损失、输入尺寸、种子及初始化来源；不同配置应另记实验，不能覆盖原结果。续训使用原完整last.pth及不变的配方，不能用推理导出替代。

完整训练及核验通过后，`python -m mine_selective.evaluate` 读取原运行目录并产生标准评估JSON与地点矩阵。要重新评估，需一并保留原best/last、config、environment、history和completion等文件，只有一个推理权重不足以通过完整运行核验。`python -m mine_relay.compare` 可比较匹配的地点矩阵。当前包只是源码搬迁检查，不代表已经在新目录重新训练过120轮。

## 推理权重

正式导出只能来自完成的120轮开发实验。导出文件与同目录 `export.json` 必须一起保存；加载器会核对文件哈希、模型来源、类别映射和推理设置，并恢复非持久保存的 `mix=1`。推理导出不可恢复优化器/随机状态，不可冒充正式测试协议下的最终模型。

加载接口为 `tools/selective_inference_bundle_20261023.py` 中的 `load_for_inference`。将本包根目录和tools加入Python搜索路径后调用，输入顺序为RGB、编码深度、有效掩膜、亮度和camera-up几何张量，必须使用冻结的数据预处理合同。它不是接受任意原始相机文件的通用部署接口。默认拒绝可丢弃两轮检查权重；软件检查必须显式允许，并不得报告为研究结果。

原始MiT实现、第三方初始化权重与MUSeg数据的归属及许可需分别保留和核对。本包未附数据/预训练权重，也未对它们作新授权。
