# coding: utf-8
"""
@author Liuchen
2018
"""

import data_tools as tools
import dnn_model as dm
import data_prepare as dp
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger('main')

# ================== step0: 定义超参数 =================
learning_rate = 0.001     # 学习率
batch_size = 128        # mini-batch大小
refine = False          # 词向量矩阵是否参与训练
epochs = 20              # 数据迭代次数
lstm_sizes = [256]  # 各层lstm的维度
embed_size = 300        # 词向量维度
keep_prob = 0.5         # drop out 保留率
max_sent_len = 240      # 最大句长
l2reg = 0.000            # l2正则化参数

class_num = 2           # 类别数量
lang = 'EN'             # 文本语言 EN为英文，CN为中文
train_percent = 0.8     # 训练数据的比例
show_step = 2          # 每隔几个批次输出一次结果

# ================== step1: 数据准备 =================
## a. 从csv文件读取数据
texts, labels = dp.load_from_csv("../data/data.csv")
## b. 从以Tab符为分隔符的csv文件读取数据
# texts, labels = dp.load_from_csv("../data/cn_data.txt", delimiter='\t', lang=lang)
## c. 从情感类别文件读取数据
# texts, labels = dp.load_from_class_files(['../data/pos.txt', '../data/neg.txt'])

# 分词（英文按空格分，中文利用hanlp分词）
texts = tools.sentences2wordlists(texts, lang=lang)

## a. 基于文本构建词典    ---不使用已有词向量
vocab_to_int, int_to_vocab = tools.make_dictionary_by_text(" ".join([" ".join(text) for text in texts]))
embedding_matrix = None  # 设词向量矩阵为None

## b. 基于词向量构建词典 ---使用已有词向量
# vocab_to_int, embedding_matrix = tools.load_embedding("../data/word_embedding_300_new.txt") # 英文词向量
# vocab_to_int, embedding_matrix = tools.load_embedding("../data/sgns.weibo.word.txt") # 中文词向量

# 利用词典，将文本句子转成id列表
texts = tools.wordlists2idlists(texts, vocab_to_int)
# 清除预处理后文本内容为空的数据
texts, labels = tools.drop_empty_texts(texts, labels)
# 将数据中类别转为one-hot向量表示
labels = tools.labels2onehot(labels, class_num)
# 在每个长度小于最大句长的句子左侧补0
texts = tools.dataset_padding(texts, sent_len=max_sent_len)
# 将数据分为训练集（80%）、开发集（10%）和测试集（10%）
train_x, train_y, val_x, val_y, test_x, test_y = tools.dataset_split(texts, labels,
                                                                     train_percent=train_percent)

# ================== step2: 构建模型 =================
vocab_size = len(vocab_to_int)   # add one for padding
model = dm.DNNModel(class_num=class_num, batch_size=batch_size,
                    embed_dim=embed_size, rnn_dims=lstm_sizes,
                    vocab_size=vocab_size, embed_matrix=embedding_matrix,
                    l2reg=l2reg, refine=refine, learning_rate=learning_rate)
model.build()

# ================== step3: 训练 =================
# with tf.Graph().as_default():
model.train(train_x, train_y, val_x, val_y, epochs, keep_prob, show_step=show_step)

# ================== step4: 测试 =================
model.test_network(test_x, test_y, batch_size, model)
