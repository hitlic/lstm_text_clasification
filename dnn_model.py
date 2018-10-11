# coding: utf-8
"""
@author Liuchen
2018
"""
import tensorflow as tf
import data_tools as dt
import numpy as np
import os
import logging
logger = logging.getLogger('main.dnn_model')


class DNNModel():
    def __init__(self, class_num, embed_dim, rnn_dims, vocab_size=None, embed_matrix=None,
                 fc_size=500, max_sent_len=200, refine=False):
        self.class_num = class_num              # 分类类别数量
        self.embed_dim = embed_dim              # 词向量维度
        self.rnn_dims = rnn_dims                # RNN隐层维度，可有多层RNN
        if vocab_size is None and embed_matrix is None:  # 词向量和词典长度必须给出一个
            raise Exception("One of vocab_size and embed_matrix must be given!")
        self.vocab_size = vocab_size            # 词典大小
        self.embed_matrix = embed_matrix        # 词向量矩阵
        self.fc_size = fc_size                  # 全连接层大小
        self.max_sent_len = max_sent_len        # 最大句长
        self.refine = refine

        # ---- 以下为 placeholder 参数
        self.learning_rate = tf.placeholder_with_default(0.01, shape=(), name='learning_rate')      # 学习率
        self.keep_prob = tf.placeholder_with_default(
            1.0, shape=(), name='keep_prob')           # dropout keep probability
        self.l2reg = tf.placeholder_with_default(0.0, shape=(), name='L2reg')               # L2正则化参数

    def inputs_layer(self):
        """
        模型输入
        :return: 数据、标记、dropout的placeholder
        """
        with tf.name_scope('input_layer'):
            self.inputs = tf.placeholder(tf.int32, [None, self.max_sent_len], name='inputs')  # 输入数据x placeholder
            self.labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')  # 输入数据y placeholder
        return self.inputs

    def embedding_layer(self, inputs_):
        """
        词向量层
        """
        with tf.name_scope("embedding_layer"):
            if self.embed_matrix is None:   # 若无已训练词向量
                embedding = tf.Variable(tf.random_uniform((self.vocab_size, self.embed_dim), -1, 1), name="embedding")
            else:                           # 若已有词向量
                embedding = tf.Variable(self.embed_matrix, trainable=self.refine, name="embedding")
            embed = tf.nn.embedding_lookup(embedding, inputs_)
        return embed

    def rnn_layer(self, embed):
        """
        RNN层
        """
        with tf.name_scope("rnn_layer"):
            embed = tf.nn.dropout(embed, keep_prob=self.keep_prob)  # dropout
            # --- 可选的RNN单元
            # tf.contrib.rnn.BasicRNNCell(size)
            # tf.contrib.rnn.BasicLSTMCell(size)
            # tf.contrib.rnn.LSTMCell(size)
            # tf.contrib.rnn.GRUCell(size, activation=tf.nn.relu)
            # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size)
            # tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(size)

            lstms = [tf.contrib.rnn.LSTMCell(size) for size in self.rnn_dims]
            #  dropout
            drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob) for lstm in lstms]
            # 组合多个 LSTM 层
            cell = tf.contrib.rnn.MultiRNNCell(drops)
            lstm_outputs, _ = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)

            outputs = lstm_outputs[:, -1]  # 返回每条数据的最后输出

        return outputs

    def fc_layer(self, inputs):
        """
        全连接层
        """
        # initializer = tf.contrib.layers.xavier_initializer()  # xavier参数初始化，暂没用到
        with tf.name_scope("fc_layer"):
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob, name='drop_out')  # dropout
            # outputs = tf.contrib.layers.fully_connected(inputs, self.fc_size, activation_fn=tf.nn.relu)
            outputs = tf.layers.dense(inputs, self.fc_size, activation=tf.nn.relu)
        return outputs

    def output_layer(self, inputs):
        """
        输出层
        """
        with tf.name_scope("output_layer"):
            inputs = tf.layers.dropout(inputs, rate=1-self.keep_prob)
            outputs = tf.layers.dense(inputs, self.class_num, activation=None)
            # outputs = tf.contrib.layers.fully_connected(inputs, self.class_num, activation_fn=None)
        return outputs

    def set_loss(self):
        """
        损失函数
        """
        # softmax交叉熵损失
        with tf.name_scope("loss_scope"):
            reg_loss = tf.contrib.layers.apply_regularization(  # L2正则化
                tf.contrib.layers.l2_regularizer(self.l2reg),
                tf.trainable_variables()
            )
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.predictions, labels=self.labels)) + reg_loss   # ---GLOBAL---损失函数
            tf.summary.scalar("loss_summary", self.loss)

    def set_accuracy(self):
        """
        准确率
        """
        with tf.name_scope("accuracy_scope"):
            correct_pred = tf.equal(tf.argmax(self.predictions, axis=1), tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   # ---GLOBAL---准确率
            self.acc_summary = tf.summary.scalar("acc_summary", self.accuracy)

    def set_optimizer(self):
        """
        优化器
        """
        with tf.name_scope("optimizer"):
            # --- 可选优化算法
            # self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss)
            # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def build(self):
        """
        DNN模型构建
        """
        inputs = self.inputs_layer()
        embedding = self.embedding_layer(inputs)
        rnn_outputs = self.rnn_layer(embedding)
        fc_outputs = self.fc_layer(rnn_outputs)
        self.predictions = self.output_layer(fc_outputs)
        self.set_loss()
        self.set_optimizer()
        self.set_accuracy()


def train(dnn_model, learning_rate, train_x, train_y, dev_x, dev_y, epochs, batch_size, keep_prob, l2reg, dev_step=10,
          checkpoint_path="./checkpoints"):
    """
    训练并验证
    :param dnn_model: 计算图模型
    :param learning_rate: 学习率
    :param train_x: 训练数据
    :param train_y: 标记训练数据
    :param dev_x: 验证数据
    :param dev_y: 标记验证数据
    :param epochs: 迭代次数
    :param batch_size: minibatch 大小
    :param keep_prob: dropout keep probability
    :param l2reg: L2 正则化系数
    :param dev_step: 隔多少步验证一次
    :param checkpoint_path: 模型保存位置
    """
    if not os.path.exists(checkpoint_path):  # 模型保存路径不存在则创建路径
        os.makedirs(checkpoint_path + '/best')

    saver = tf.train.Saver()
    best_acc = 0

    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        train_log_writer = tf.summary.FileWriter("./log", sess.graph)
        dev_log_writer = tf.summary.FileWriter("./log/dev")

        sess.run(tf.global_variables_initializer())
        n_batches = len(train_x)//batch_size
        step = 0
        for e in range(epochs):
            train_acc = []
            dev_acc_epoch = []
            for id_, (x, y) in enumerate(dt.make_batches(train_x, train_y, batch_size), 1):
                step += 1
                feed = {
                    dnn_model.inputs: x,
                    dnn_model.labels: y,
                    dnn_model.learning_rate: learning_rate,
                    dnn_model.keep_prob: keep_prob,
                    dnn_model.l2reg: l2reg
                }
                train_loss, _, batch_acc, train_summary = sess.run(
                    [dnn_model.loss, dnn_model.optimizer, dnn_model.accuracy, merged_summary],
                    feed_dict=feed,
                )
                train_acc.append(batch_acc)

                train_log_writer.add_summary(train_summary, step)  # 写入日志

                # 验证 ------------
                if step % dev_step == 0:
                    dev_acc = []
                    for xx, yy in dt.make_batches(dev_x, dev_y, batch_size):
                        feed = {
                            dnn_model.inputs: xx,
                            dnn_model.labels: yy,
                            dnn_model.keep_prob: 1,
                        }
                        dev_loss, dev_batch_acc, dev_acc_summary = sess.run(
                            [dnn_model.loss, dnn_model.accuracy, dnn_model.acc_summary],
                            feed_dict=feed
                        )
                        dev_acc.append(dev_batch_acc)
                        dev_acc_epoch.append(dev_batch_acc)
                        dev_log_writer.add_summary(dev_acc_summary, step)

                    info = "|Epoch {}/{}\t".format(e+1, epochs) + \
                        "|Batch {}/{}\t".format(id_+1, n_batches) + \
                        "|Train-Loss| {:.5f}  ".format(train_loss) + \
                        "|Dev-Loss| {:.5f}  ".format(dev_loss) + \
                        "|Train-Acc| {:.5f}  ".format(np.mean(train_acc)) + \
                        "|Dev-Acc| {:.5f}".format(np.mean(dev_acc))
                    logger.info(info)

            # global_step 用于生成多个checkpoint的文件名
            # saver.save(sess, checkpoint_path+"model.ckpt", global_step=e)  # 保存每个epoch的模型

            # 选择最好的模型保存
            avg_dev_acc = np.mean(dev_acc_epoch)
            if best_acc < avg_dev_acc:
                best_acc = avg_dev_acc
                saver.save(sess, checkpoint_path+"/best/best_model.ckpt")
        logger.info("** The best dev accuracy: {:.5f}".format(best_acc))


def test(dnn_model, test_x, test_y, batch_size, model_dir="./checkpoints/best"):
    """
    利用最好的模型进行测试
    :param test_x: 测试数据
    :param test_y: 标记测试数据
    :param batch_size: 批次大小
    :param dnn_model: 原dnn模型
    :param model_dir: 训练好的模型的存储位置
    """

    saver = tf.train.Saver()
    test_acc = []
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        for _, (x, y) in enumerate(dt.make_batches(test_x, test_y, batch_size), 1):
            feed = {dnn_model.inputs: x,
                    dnn_model.labels: y,
                    dnn_model.keep_prob: 1,
                    }
            batch_acc = sess.run([dnn_model.accuracy], feed_dict=feed)
            test_acc.append(batch_acc)
        logger.info("** Test Accuracy: {:.5f}".format(np.mean(test_acc)))
