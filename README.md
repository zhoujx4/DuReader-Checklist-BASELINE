# 项目说明:
百度2021年语言与智能技术竞赛机器阅读理解Pytorch版baseline  
比赛链接:https://aistudio.baidu.com/aistudio/competition/detail/66?isFromLuge=true  
> 官方的baseline版本是基于paddlepaddle框架的,我把它改写成了Pytorch框架,其中大部分代码沿用的是官方提供的代码,只是有一些框架部分进行了修改,另外增加了早停策略/对抗训练等优化措施,习惯用Pytorch版本的可以基于此进行优化.

# 环境
- python=3.6
- torch=1.7
- transformers=4.5.0

# 训练示例
训练  
```
python run.py
--max_len=256
--model_name_or_path=下载的预训练模型路径
--per_gpu_train_batch_size=7
--per_gpu_eval_batch_size=40
--learning_rate=1e-5
--linear_learning_rate=1e-4
--num_train_epochs=100
--output_dir="./output"
--weight_decay=0.01
--early_stop=2
```

预测
```
python predict.py
--max_len=400
--model_name_or_path=下载的预训练模型路径
--per_gpu_eval_batch_size=120
--output_dir="./output"
--fine_tunning_model=微调后的模型路径
```
# 实验结果
用的baseline模型是base版MacBERT(具体请看https://github.com/ymcui/MacBERT)

![image-20210410231128986](https://raw.githubusercontent.com/zhoujx4/PicGo/main/img/image-20210410231128986.png)

# 后续优化策略
- 数据清洗，据官方工作人员讲解到，训练集的准确率只能确保92%以上
- 更多的数据
- 更细粒度的数据增强
- 模型结构的优化