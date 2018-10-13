# lstm_text_clasification
 一个基本的多层lstm rnn模型，能实现中英文文本的二分类或多分类。

### 运行环境
- python 3.6

### 软件包依赖
- tensorflow 1.6以及

- pandas

- pyhanlp (若处理中文数据时必须，要求jdk安装配置好)

### 词向量
- data文件夹中的英文词向量维度为300，但词汇量太小，仅做测试用。可从这里下载替换 http://nlp.stanford.edu/data/glove.6B.zip
- 中文词向量参见  https://github.com/Embedding/Chinese-Word-Vectors

### 功能说明
- 一个基本的多层lstm rnn模型，能实现中英文文本的二分类或多分类
- 能够保存训练过程中最佳的模型，用于测试。（也可保存每个epoch的模型，去掉dnn_model.py中train方法相应部分代码的注释即可）
- 能够输出日志，包括计算图，以及loss、train_accuracy、dev_accuracy，可利用tensorboard查看。（也可输出元日志，去除中train方法相应部分代码的注释即可，但日志文件相当巨大，且会明显影响训练速度，建议在必要时再打开）

### 拓展
- 若要调参，修改train.py中的超参数设置部分
- 若数据格式有变，修改data_prepare.py，读取新格式的数据
- 若模型有变，修改dnn_model.py，修改或添加自己设计好的layer，并在build方法中加入相应代码
