# coding: utf-8
"""
@author Liuchen
2018
"""
import tensorflow as tf
import data_tools as dt
import numpy as np
import logging
logger = logging.getLogger('main.dnn_model')


class DNNModel():
    def __init__(self, class_num, embed_dim, rnn_dims, batch_size=256,
                 vocab_size=None, embed_matrix=None, l2reg=0.0001, refine=False,
                 fc_size=2000, decay_step=200, decay_rate=0.95):
        self.class_num = class_num              # 分类类别数量
        self.embed_dim = embed_dim              # 词向量维度
        self.rnn_dims = rnn_dims                # RNN隐层维度，可有多层RNN
        self.batch_size = batch_size            # 批次大小
        if vocab_size is None and embed_matrix is None:  # 词向量和词典长度必须给出一个
            raise Exception("One of vocab_size and embed_matrix must be given!")
        self.vocab_size = vocab_size            # 词典大小
        self.embed_matrix = embed_matrix        # 词向量矩阵
        self.l2reg = l2reg                      # l2正则化参数
        self.refine = refine                    # 使用已有词向量时，词向量是否参与训练
        self.fc_size = fc_size                  # 全连接层大小
        self.decay_step = decay_step            # 学习率衰减步
        self.decay_rate = decay_rate            # 学习率的衰减率

    def inputs_layer(self):
        """
        模型输入
        :return: 数据、标记、dropout的placeholder
        """
        keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')
        with tf.name_scope('input_layer'):
            inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
            labels_ = tf.placeholder(tf.int32, [None, self.class_num], name='labels')
        self.inputs = inputs_        # ---GLOBAL--- 输入数据x placeholder
        self.labels = labels_        # ---GLOBAL--- 输入数据y placeholder
        self.keep_prob = keep_prob_  # ---GLOBAL--- dropout keep probability
        return inputs_

    def embedding_layer(self, inputs_):
        """
        词向量层
        """
        with tf.name_scope("embedding_layer"):
            if self.embed_matrix is None:   # 若无已训练词向量
                embedding = tf.Variable(tf.random_uniform((self.vocab_size, self.embed_dim), -1, 1), name="embedding")
            else:                           # 若已有词向量
                embedding = tf.Variable(self.embed_matrix, trainable=self.refine, name="embedding")
            embed = tf.nn.dropout(tf.nn.embedding_lookup(embedding, inputs_), keep_prob=self.keep_prob)
        return embed

    def rnn_layer(self, embed):
        """
        RNN层
        """
        with tf.name_scope("rnn_layer"):
            # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
            lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.rnn_dims]
            #  dropout
            drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob) for lstm in lstms]
            # 组合多个 LSTM 层
            cell = tf.contrib.rnn.MultiRNNCell(drops)
            # 初始装态置0
            initial_state = cell.zero_state(self.batch_size, tf.float32)

            lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
        self.rnn_cell = cell                    # ---GLOBAL--- rnn 单元格
        self.rnn_initial_state = initial_state  # ---GLOBAL--- rnn初始状态
        self.rnn_final_state = final_state      # ---GLOBAL--- rnn 最终状态

        return lstm_outputs[:, -1]  # 返回每个数据最后输出

    def fc_layer(self, inputs):
        """
        全连接层
        """
        # initializer = tf.contrib.layers.xavier_initializer()  # xavier参数初始化，暂没用到
        with tf.name_scope("FC_layer"):
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob, name='drop_out')  # dropout
            outputs = tf.contrib.layers.fully_connected(inputs, self.fc_size, activation_fn=tf.nn.relu)
        return outputs

    def output_layer(self, inputs):
        """
        输出层
        """
        with tf.name_scope("output_layer"):
            outputs = tf.contrib.layers.fully_connected(inputs, self.class_num, activation_fn=None)
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
            correct_pred = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.labels, 1))
            # correct_pred = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   # ---GLOBAL---准确率
            self.acc_summary = tf.summary.scalar("acc_summary", self.accuracy)

    def build(self):
        """
        DNN模型构建
        """
        inputs = self.inputs_layer()
        embedding = self.embedding_layer(inputs)
        rnn_outputs = self.rnn_layer(embedding)
        fc_outputs = self.fc_layer(rnn_outputs)
        predictions = self.output_layer(fc_outputs)
        self.predictions = predictions    # ---GLOBAL--- 预测结果
        self.set_loss()
        self.set_accuracy()

    def set_optimizer(self, learning_rate, global_step):
        """
        优化器
        """
        with tf.name_scope("optimizer"):
            # self.optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(self.loss, global_step=global_step)  # ---GLOBAL--- 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

    def train(self, train_x, train_y, dev_x, dev_y, learning_rate, epochs, keep_prob,
              checkpoint_path="./checkpoints/", show_step=10):
        """
        训练并验证
        """
        saver = tf.train.Saver()
        best_acc = 0

        global_step = tf.Variable(0, trainable=False, name='global_step')
        learn_rate_decay = tf.train.exponential_decay(                  # ---- 学习率衰减
            learning_rate, global_step, self.decay_step, self.decay_rate, staircase=True)
        self.set_optimizer(learn_rate_decay, global_step)

        merged_summary = tf.summary.merge_all()

        with tf.Session() as sess:
            train_log_writer = tf.summary.FileWriter("./log", sess.graph)
            dev_log_writer = tf.summary.FileWriter("./log/dev")
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  # for meta日志
            run_metadata = tf.RunMetadata()  # for meta日志

            sess.run(tf.global_variables_initializer())
            n_batches = len(train_x)//self.batch_size
            step = 0
            for e in range(epochs):
                init_state = sess.run(self.rnn_initial_state)

                train_acc = []
                dev_acc_epoch = []
                for id_, (x, y) in enumerate(dt.make_batches(train_x, train_y, self.batch_size), 1):
                    step += 1
                    feed = {
                        self.inputs: x,
                        self.labels: y,
                        self.keep_prob: keep_prob,
                        self.rnn_initial_state: init_state
                    }
                    loss_, _, _, _, batch_acc, train_summary = sess.run(
                        [self.loss, self.rnn_final_state, self.optimizer, learn_rate_decay, self.accuracy, merged_summary],
                        feed_dict=feed,
                        options=run_options,       # for meta 日志
                        run_metadata=run_metadata  # for meta 日志
                    )
                    train_acc.append(batch_acc)

                    # # ------写入meta日志，若打开日志文件会特别巨大
                    # train_log_writer.add_run_metadata(run_metadata, 'batch%03d' % step)
                    train_log_writer.add_summary(train_summary, step)  # 写入日志

                    # 验证 ------------
                    if id_ % show_step == 0:
                        dev_acc = []
                        dev_state = sess.run(self.rnn_cell.zero_state(self.batch_size, tf.float32))
                        for xx, yy in dt.make_batches(dev_x, dev_y, self.batch_size):
                            feed = {
                                self.inputs: xx,
                                self.labels: yy,
                                self.keep_prob: 1,
                                self.rnn_initial_state: dev_state
                            }
                            dev_batch_acc, dev_state, dev_acc_summary = sess.run(
                                [self.accuracy, self.rnn_final_state, self.acc_summary],
                                feed_dict=feed
                            )
                            dev_acc.append(dev_batch_acc)
                            dev_acc_epoch.append(dev_batch_acc)
                            dev_log_writer.add_summary(dev_acc_summary, step)

                        info = "|Epoch {}/{}\t".format(e+1, epochs) + \
                            "|Batch {}/{}\t".format(id_+1, n_batches) + \
                            "|Loss| {:.5f}  ".format(loss_) + \
                            "|Train-Acc| {:.5f}  ".format(np.mean(train_acc)) + \
                            "|Dev-Acc| {:.5f}".format(np.mean(dev_acc))
                        logger.info(info)

                # global_step 用于生成多个checkpoint的文件名
                # saver.save(sess, checkpoint_path+"model.ckpt", global_step=e)  # 保存每个epoch的模型

                # 选择最好的模型保存
                avg_dev_acc = np.mean(dev_acc_epoch)
                if best_acc < avg_dev_acc:
                    best_acc = avg_dev_acc
                    saver.save(sess, checkpoint_path+"best/best_model.ckpt")
            logger.info("** The best dev accuracy: {:.5f}".format(avg_dev_acc))

    @staticmethod
    def test_network(test_x, test_y, batch_size, dnn_model, model_dir="./checkpoints/best"):
        """
        利用最好的模型进行测试
        :param test_x: x测试数据
        :param test_y: 标记测试数据
        :param batch_size: 批次大小
        :param dnn_model: 原dnn模型
        :param model_dir: 训练好的模型的存储位置
        """
        model = dnn_model

        saver = tf.train.Saver()

        test_acc = []
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            test_state = sess.run(model.rnn_cell.zero_state(batch_size, tf.float32))
            for _, (x, y) in enumerate(dt.make_batches(test_x, test_y, batch_size), 1):
                feed = {model.inputs: x,
                        model.labels: y,
                        model.keep_prob: 1,
                        model.rnn_initial_state: test_state}
                batch_acc, test_state = sess.run([model.accuracy, model.rnn_final_state], feed_dict=feed)
                test_acc.append(batch_acc)
            logger.info("** Test Accuracy: {:.5f}".format(np.mean(test_acc)))
