# 一个基于seq2seq模型，由pytorch官方教程改进而来的对话机器人



## 运行环境要求

python 3.x

torch 1.10.0+cu111

如果仍然缺少环境请使用

```
pip install + 包名
```

命令安装相应的依赖包即可



## config.py 

配置文件，包含模型的各种训练参数。

NOTE： 模型参数已经经过多次调参调优，非实验环境下不建议也没必要再额外更改模型参数



## data_util.py

数据处理文件，使用时不需要进行更改，数据集来源康奈尔电影对话集



## seq2seq.py

核心的模型文件，包含Encoder编码器和Decoder解码器，并使用了Loung注意力机制



## train.py

训练模型文件，在python3环境下直接输入

```
python train.py
```

进行训练，训练过程中请务必开启GPU加速



## predict.py

与机器人对话的运行文件，在训练完成后执行

```
python predict.py
```

就可以与机器人愉快进行对话了



## 后续版本的升级计划 TODO：

1、增强机器人的长时记忆能力

2、升级为中文聊天机器人