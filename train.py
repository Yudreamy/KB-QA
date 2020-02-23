import os
import codecs               # 用作编码转换
import time
import sys
import numpy as np
import tensorflow as tf
import data_util
import similarity
from biLSTM import BiLSTM

# --------------Parameters begin--------------

# data loading params
tf.flags.DEFINE_string("knowledge_file", "data/knowledge.txt", "Knowledge data.")
tf.flags.DEFINE_string("train_file", "data/train.txt", "Training data.")
tf.flags.DEFINE_string("test_file", "data/test.txt", "Test data.")
tf.flags.DEFINE_string("stop_words_file", "data/stop_words.txt", "Stop words.")

# result & model save params
tf.flags.DEFINE_string("result_file", "res/predictRst.score", "Predict result.")
tf.flags.DEFINE_string("save_file", "res/savedModel", "Save model.")

# pre-trained word embedding vectors 载入的词向量是基于wiki百科中文语料训练出来的word2vec，可直接下载到
# Path to embedding file!
tf.flags.DEFINE_string("embedding_file", "D:/KB-QA-master/zhwiki_2017_03.sg_50d.word2vec", "Embedding vectors.")

# hyperparameters 超参数
tf.flags.DEFINE_integer("k", 5, "K most similarity knowledge (default: 5).")        # 取最相关的 5 个 knowledge
tf.flags.DEFINE_integer("rnn_size", 100, "Neurons number of hidden layer in LSTM cell (default: 100).")     # 隐藏层神经元数
tf.flags.DEFINE_float("margin", 0.1, "Constant of max-margin loss (default: 0.1).")         # max-margin，参考 Hinge loss，我们希望正样本分数越高越好，负样本分数越低越好，但不希望两者分数差距太大，设margin
tf.flags.DEFINE_integer("max_grad_norm", 5, "Control gradient expansion (default: 5).")     # 梯度的最大范数，防止梯度爆炸
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50).")        # 嵌入的维度
tf.flags.DEFINE_integer("max_sentence_len", 100, "Maximum number of words in a sentence (default: 100).")   # 一个句子中最大单词数
tf.flags.DEFINE_float("dropout_keep_prob", 0.45, "Dropout keep probability (default: 0.5).")        # 训练时的 dropout
tf.flags.DEFINE_float("learning_rate", 0.4, "Learning rate (default: 0.4).")                        # 学习率
tf.flags.DEFINE_float("lr_down_rate", 0.5, "Learning rate down rate(default: 0.5).")                # 每次减少学习率的因子
tf.flags.DEFINE_integer("lr_down_times", 4, "Learning rate down times (default: 4)")                # 减少学习率的次数
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")              # L2 正则化参数，防止过拟合

# training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")                      # 一次训练选取的样本数 https://blog.csdn.net/qq_34886403/article/details/82558399
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")        # epoch（数据集中的所有样本都跑过一遍） 数
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")     # 每 50 step 评估一下准确度
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")     # 每 100 step 保存模型
tf.flags.DEFINE_integer("num_checkpoints", 20, "Number of checkpoints to store (default: 5)")           # 保存检查点个数， https://www.jianshu.com/p/adfa8aa2cbdf

# gpu parameters
tf.flags.DEFINE_float("gpu_mem_usage", 0.75, "GPU memory max usage rate (default: 0.75).")  # GPU 内存最大使用率
tf.flags.DEFINE_string("gpu_device", "/gpu:0", "GPU device name.")          # GPU 设备名称

# misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement.")    # 要不要自动分配

FLAGS = tf.flags.FLAGS          # 这块就是打印一下相关初始参数
"""
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
"""
# --------------Parameters end--------------


# load pre-trained embedding vector
print("loading embedding...")
embedding, word2idx = data_util.load_embedding(FLAGS.embedding_file)        # 初始 embedding 向量，单词索引

# load stop words
stop_words = codecs.open(FLAGS.stop_words_file, 'r', encoding='utf8').readlines()
stop_words = [w.strip() for w in stop_words]                                # 所有停用词列表

# top k most related knowledge , 进入 similarity.py 文件
print("computing similarity...")
similarity.generate_dic_and_corpus(FLAGS.knowledge_file, FLAGS.train_file, stop_words)      # 形成 knowledge_file,train_file 分词结合之后的词典
train_sim_ixs = similarity.topk_sim_ix(FLAGS.train_file, stop_words, FLAGS.k)               # train 的逐条 topk 最相似 knowledge 
test_sim_ixs = similarity.topk_sim_ix(FLAGS.test_file, stop_words, FLAGS.k)                 # test 的逐条 topk 最相似 knowledge


# --------------Data preprocess begin--------------
print("loading data...")            # 见 data_util.py，构成训练集问题答案组，及测试集问题答案组
train_questions, train_answers, train_labels, train_question_num = \
    data_util.load_data(FLAGS.knowledge_file, FLAGS.train_file, word2idx, stop_words, train_sim_ixs, FLAGS.max_sentence_len)

test_questions, test_answers, test_labels, test_question_num = \
    data_util.load_data(FLAGS.knowledge_file, FLAGS.test_file, word2idx, stop_words, test_sim_ixs, FLAGS.max_sentence_len)

#print(train_question_num, len(train_questions), len(train_answers), len(train_labels))
#print(test_question_num, len(test_questions), len(test_answers), len(test_labels))


# 正确选项分别与该问题中所有错误选项组合，构成三个答案组合，分别与大问题组合构成三个样例
questions, true_answers, false_answers = [], [], []
for q, ta, fa in data_util.training_batch_iter(
        train_questions, train_answers, train_labels, train_question_num, FLAGS.batch_size):
    questions.append(q), true_answers.append(ta), false_answers.append(fa)
# --------------Data preprocess end--------------


# --------------Training begin--------------
print("training...")
with tf.Graph().as_default(), tf.device(FLAGS.gpu_device):      # Tensorflow 程序通过 tf.device 函数来指定运行每一个操作的设备
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.gpu_mem_usage     # 限制 GPU 的使用率，占 gpu_mem_usage（75%）显存，https://blog.csdn.net/lty_sky/article/details/91491302
    )
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,        # 如果指定的设备不存在，允许 Tensorflow 自动分配设备
        gpu_options=gpu_options
    )
    with tf.Session(config=session_conf).as_default() as sess:
        globalStep = tf.Variable(0, name="global_step", trainable=False)            # 全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等，https://blog.csdn.net/leviopku/article/details/78508951
        lstm = BiLSTM(              # 双向 LSTM，biLSTM.py
            FLAGS.batch_size,
            FLAGS.max_sentence_len,
            embedding,
            FLAGS.embedding_dim,
            FLAGS.rnn_size,
            FLAGS.margin
        )
        
        # define training procedure，训练过程
        tvars = tf.trainable_variables()            # 获取所有可训练的向量
        grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), FLAGS.max_grad_norm)      # 计算向量梯度
        saver = tf.train.Saver()                    # 保存模型

        # output directory for models and summaries
        #timestamp = str(int(time.time()))           # 时间戳
        #out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))      # 存在当前目录下的 runs 文件夹中
        #print("Writing to {}\n".format(out_dir))

        print("Writing to {}\n".format(os.getcwd()+r"\summary"))
        # summaries
        loss_summary = tf.summary.scalar("loss", lstm.loss)     # tf.summary.scalar 用来显示标量信息,一般在画 loss,accuary 时会用到这个函数
        summary_op = tf.summary.merge([loss_summary])           # tf.summary.merge 方法有选择性地保存信息

        #summary_dir = os.path.join(out_dir, "summary", "train")
        summary_writer = tf.summary.FileWriter('summary', sess.graph)     # 保存图到指定文件夹，通过 TensorBoard 可实现可视化

        # evaluating
        def evaluate():
            print("evaluating..")
            scores = []
            for test_q, test_a in data_util.testing_batch_iter(test_questions, test_answers, test_question_num, FLAGS.batch_size):
                test_feed_dict = {      # 得到每一批的问题集和答案集
                    lstm.inputTestQuestions: test_q,
                    lstm.inputTestAnswers: test_a,
                    lstm.dropout_keep_prob: 1.0         # 测试时不用 dropout
                }
                _, score = sess.run([globalStep, lstm.result], test_feed_dict)      # 运行，result -> 测试集上 (q,a) 的余弦相似度，https://blog.csdn.net/baidu_15113429/article/details/86530218
                scores.extend(score)
            cnt = 0         # 记录预测成功的个数
            scores = np.absolute(scores)    # np.absolute 取绝对值
            for test_id in range(test_question_num):        # 遍历所有测试问题
                offset = test_id * 4
                predict_true_ix = np.argmax(scores[offset:offset + 4])      # 4 个答案中最后分数最大的那个作为正确结果
                if test_labels[offset + predict_true_ix] == 1:              # 如果对应标签为 1
                    cnt += 1                                                # 猜测正确 +1
            print("evaluation acc: ", cnt / test_question_num)              # 打印测试准确度

            # 以相同方法计算训练集上的准确度，代码相同，不再赘述
            scores = []
            for train_q, train_a in data_util.testing_batch_iter(train_questions, train_answers, train_question_num, FLAGS.batch_size):
                test_feed_dict = {
                    lstm.inputTestQuestions: train_q,
                    lstm.inputTestAnswers: train_a,
                    lstm.dropout_keep_prob: 1.0
                }
                _, score = sess.run([globalStep, lstm.result], test_feed_dict)
                scores.extend(score)
            cnt = 0
            scores = np.absolute(scores)
            for train_id in range(train_question_num):
                offset = train_id * 4
                predict_true_ix = np.argmax(scores[offset:offset + 4])
                if train_labels[offset + predict_true_ix] == 1:
                    cnt += 1
            print("evaluation acc(train): ", cnt / train_question_num)

        # training
        sess.run(tf.global_variables_initializer())
        lr = FLAGS.learning_rate
        for i in range(FLAGS.lr_down_times):
            optimizer = tf.train.GradientDescentOptimizer(lr)       # 梯度下降优化，指定学习率
            optimizer.apply_gradients(zip(grads, tvars))
            trainOp = optimizer.apply_gradients(zip(grads, tvars), global_step=globalStep)  # 将梯度应用于变量
            for epoch in range(FLAGS.num_epochs):
                for question, trueAnswer, falseAnswer in zip(questions, true_answers, false_answers):
                    feed_dict = {           # 初始化 (q,a+,a-)
                        lstm.inputQuestions: question,
                        lstm.inputTrueAnswers: trueAnswer,
                        lstm.inputFalseAnswers: falseAnswer,
                        lstm.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    }
                    _, step, _, _, loss, summary = \
                        sess.run([trainOp, globalStep, lstm.trueCosSim, lstm.falseCosSim, lstm.loss, summary_op], feed_dict)    # 运行
                    print("step:", step, "loss:", loss)         # 打印
                    summary_writer.add_summary(summary, step)
                    if step % FLAGS.evaluate_every == 0:        # 每 50 step 开始评估
                        evaluate()

                saver.save(sess, FLAGS.save_file)
            lr *= FLAGS.lr_down_rate        # 减小学习率

        # final evaluate
        evaluate()
# --------------Training end--------------
