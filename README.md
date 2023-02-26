### 环境配置

Python 版本：3.9.12
PyTorch 版本：1.11.0
CUDA 版本：11.3

所需环境在 `requirements.txt` 中定义。

### 预训练模型

* 使用了 huggingface 上提供的 `hfl/chinese-macbert-base` 模型。链接为： https://huggingface.co/hfl/chinese-macbert-base

### 算法描述

* 分别建立双流以及创新的混流模型。  双流：双支路分别提取文本图像特征；混流：引入早期融合机制，兼顾单双流优点，并创建了Xlarge,large,small版本
* 对百万无标签数据进行掩码文本 mlm、图文匹配 itm 预训练，mask比率0.25

### 性能

B榜测试性能：

large，double：0.697左右；

large，small，double：0.700左右；

large，small，double，Xlarge：0.702左右（预训练时间太长，不放差不多也能到误差以内）

有问题联系：18361226138
