# lstm_text_clasification
 一个基本的多层lstm rnn模型，能实现中英文文本的二分类或多分类

### 运行环境
- python 3.6

### 软件包依赖
- tensorflow 1.6以上
- pandas
- pyhanlp (要求jdk8已安装好)

### 功能说明
1. 一个基本的多层lstm rnn模型，能实现中英文文本的二分类或多分类。
2. 能够保存训练过程中最佳的模型，用于测试。（也可保存每个epoch的模型，去掉dnn_model.py中train方法相应部分代码的注释即可）。
3. 能够输出日志，包括计算图，以及loss、train_accuracy、dev_accuracy，可利用tensorboard查看。（也可输出元日志，去除中train方法相应部分代码的注释即可，但日志文件相当巨大，建议在必要时再打开）。
4. 若要使用中文词向量，请自行下载  https://github.com/Embedding/Chinese-Word-Vectors

### 拓展
1. 若要调参，修改train.py中的超参数设置部分。
2. 若数据有变，修改data_prepare.py，读取新格式的数据；
3. 若深度学习模型有变，修改dnn_model.py，修改或添加自己设计好的layer，并在build方法中加入相应代码；
