import tensorflow as tf


class BiLSTM(object):
    def __init__(self, batch_size, max_sentence_len, embeddings, embedding_size, rnn_size, margin):     # 参数初始化
        self.batch_size = batch_size        # 一次训练选取的样本数
        self.max_sentence_len = max_sentence_len
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.margin = margin

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.inputQuestions = tf.placeholder(tf.int32, shape=[None, self.max_sentence_len])         # placeholder 占位符，执行时传入值
        self.inputTrueAnswers = tf.placeholder(tf.int32, shape=[None, self.max_sentence_len])
        self.inputFalseAnswers = tf.placeholder(tf.int32, shape=[None, self.max_sentence_len])
        self.inputTestQuestions = tf.placeholder(tf.int32, shape=[None, self.max_sentence_len])
        self.inputTestAnswers = tf.placeholder(tf.int32, shape=[None, self.max_sentence_len])

        # embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            tf_embedding = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")      # 导入 embeddings，参数可训练
            questions = tf.nn.embedding_lookup(tf_embedding, self.inputQuestions)       # tf.nn.embedding_lookup 根据 inputs_ids 中的 id，寻找 embeddings 中的第 id 行，组成一个 tensor 返回。https://blog.csdn.net/laolu1573/article/details/77170407
            true_answers = tf.nn.embedding_lookup(tf_embedding, self.inputTrueAnswers)
            false_answers = tf.nn.embedding_lookup(tf_embedding, self.inputFalseAnswers)

            test_questions = tf.nn.embedding_lookup(tf_embedding, self.inputTestQuestions)
            test_answers = tf.nn.embedding_lookup(tf_embedding, self.inputTestAnswers)
        # LSTM，布置网络
        with tf.variable_scope("LSTM_scope", reuse=None):       # variable_scope 变量作用域，实现共享变量，用于生成上下文管理器，创建命名空间
            question1 = self.biLSTMCell(questions, self.rnn_size)           # 调用下边的 biLSTMCell 函数
            question2 = tf.nn.tanh(self.max_pooling(question1))             # 激活函数
        with tf.variable_scope("LSTM_scope", reuse=True):       # reuse，命名空间可用 tf.get_variable() 获取变量
            true_answer1 = self.biLSTMCell(true_answers, self.rnn_size)
            true_answer2 = tf.nn.tanh(self.max_pooling(true_answer1))
            false_answer1 = self.biLSTMCell(false_answers, self.rnn_size)
            false_answer2 = tf.nn.tanh(self.max_pooling(false_answer1))

            test_question1 = self.biLSTMCell(test_questions, self.rnn_size)
            test_question2 = tf.nn.tanh(self.max_pooling(test_question1))
            test_answer1 = self.biLSTMCell(test_answers, self.rnn_size)
            test_answer2 = tf.nn.tanh(self.max_pooling(test_answer1))

        self.trueCosSim = self.get_cosine_similarity(question2, true_answer2)       # 计算 (q,a+) 的余弦相似度
        self.falseCosSim = self.get_cosine_similarity(question2, false_answer2)     # 计算 (q,a-) 的余弦相似度
        self.loss = self.get_loss(self.trueCosSim, self.falseCosSim, self.margin)   # 计算 loss

        self.result = self.get_cosine_similarity(test_question2, test_answer2)      # 测试集上 (q,a) 的余弦相似度

    def biLSTMCell(self, x, hidden_size):       # x.shape [batch_size, time_steps, input_size]
        input_x = tf.transpose(x, [1, 0, 2])    # 输入的数据格式转换
        input_x = tf.unstack(input_x)
        # 定义 LSTM cell，单层 LSTM
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)          # 前向 LSTM 层
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=self.dropout_keep_prob,      # dropout
                                                     output_keep_prob=self.dropout_keep_prob)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)          # 后向 LSTM 层
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=self.dropout_keep_prob,      # dropout
                                                     output_keep_prob=self.dropout_keep_prob)
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)   # 构建双向 LSTM 网络
        output = tf.stack(output)       # output_bw.shape = [time_steps, batch_size, hidden_size]
        output = tf.transpose(output, [1, 0, 2])
        return output

    @staticmethod                       # 静态方法
    def get_cosine_similarity(q, a):    # 计算余弦相似度
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))   # tf.reduce_sum()，用于计算张量沿着某一维度的和，可以在求和后降维
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cosSim = tf.div(mul, tf.multiply(q1, a1))   # 基本公式
        return cosSim

    @staticmethod
    def max_pooling(lstm_out):      # 池化，矩阵降维，每块取最大值
        height = int(lstm_out.get_shape()[1])
        width = int(lstm_out.get_shape()[2])
        lstm_out = tf.expand_dims(lstm_out, -1)     # 在最后增加一维
        output = tf.nn.max_pool(lstm_out, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output

    @staticmethod
    def get_loss(trueCosSim, falseCosSim, margin):      # Hinge Loss
        zero = tf.fill(tf.shape(trueCosSim), 0.0)
        tfMargin = tf.fill(tf.shape(trueCosSim), margin)
        with tf.name_scope("loss"):
            # max-margin losses L= max{ 0, margin-cos(V_Q,V_A+)+cos(V_Q,V_A-) }
            losses = tf.maximum(zero, tf.subtract(tfMargin, tf.subtract(trueCosSim, falseCosSim)))
            loss = tf.reduce_sum(losses)
        return loss
