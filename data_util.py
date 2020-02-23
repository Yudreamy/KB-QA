import re
from collections import defaultdict     # defaultdict 区别于 dict，当词典出现未定义的 key 时会默认返回一个根据 method_factory 参数不同的默认值
import jieba.posseg                     # jieba 分词，词性标注
import numpy as np

# 分词（with 词性），去除停用词 and 只出现一次的词
def tokenizer(filename, stop_words):
    texts = []
    with open(filename, 'r',encoding="utf-8") as f:
        for line in f.readlines():      # jiaba.cut(sentence, cut_all=False, HMM=True)切词，需要词性时使用 jieba.posseg.cut(sentence)
            texts.append([token for token, _ in jieba.posseg.cut(line.rstrip())     # str.rstrip([chars]) 删除 str 字符串末尾的指定字符（默认则为空格）
                          if token not in stop_words])          # 去除停用词表中的词

    # remove words that appear only once，去掉只出现一次的 word
    frequency = defaultdict(int)    # 初始化，frequency 记录词典中词的频率
    for text in texts:              # 如果 token 在 defaultdict 不存在，则访问它时会返回 0，之后每找到一个则会加 1
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]       # 更新 texts 列表
    return texts


def load_embedding(filename):
    embeddings = []
    word2idx = defaultdict(list)    # {单词：数字}，用于将训练数据中的单词变成数字
    with open(filename, mode="r", encoding="utf-8") as rf:
        for line in rf:             # "word embedding"
            arr = line.split(" ")
            embedding = [float(val) for val in arr[1: -1]]
            word2idx[arr[0]] = len(word2idx)
            embeddings.append(embedding)    # 嵌入矩阵，用于 embedding 的初始化

    return embeddings, word2idx


# 将单词串列表转换为索引列表
def words_list2index(words_list, word2idx,  max_len):
    """
    word list to indexes in embeddings.
    """
    # dict.get(key, default=None)，key：字典中要查找的键；default：如果指定键的值不存在时，返回该默认值
    unknown = word2idx.get("UNKNOWN", 0)
    num = word2idx.get("NUM", len(word2idx))
    index = [unknown] * max_len         # [0]*3=[0,0,0]，列表乘法，index 初始为全 0
    i = 0
    for word in words_list:
        if word in word2idx:            # 在 word2idx 中则直接转换为索引
            index[i] = word2idx[word]
        else:
            if re.match("\d+", word):   # 若 word 为一个或多个数字字符，\d 匹配数字字符[0-9]
                index[i] = num          # 对应 "NUM" 索引
            else:
                index[i] = unknown      # 对应 "UNKNOWN" 索引
        if i >= max_len - 1:            # 遍历到最大句长
            break
        i += 1
    return index


# 加载训练或测试文件，记录 questions（knowledge + B:... + Q:...）列表，答案列表（R and W），标签及总问题数
# (knowledge_file文件，文件名，单词索引，停用词，topk knowledge，句子最大长度)
def load_data(knowledge_file, filename, word2idx, stop_words, sim_ixs, max_len):
    knowledge_texts = tokenizer(knowledge_file, stop_words)     # knowledge 文件分词
    train_texts = tokenizer(filename, stop_words)               # train 文件分词

    question_num = 0                            # 记录问题数
    tmp = []                                    # 记录 knowledge + B:... + Q:...
    questions, answers, labels = [], [], []
    with open(filename, mode="r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 6 == 0:                      # 对应 B:...
                question_num += 1
                for j in sim_ixs[i//6]:         # "//"，地板除，5//2=2
                    tmp.extend(knowledge_texts[j])      # 记录跟此条相关的 knowledge
                tmp.extend(train_texts[i])              # 记录此条背景 B:...
            elif i % 6 == 1:                    # 对应 Q:...
                tmp.extend(train_texts[i])      # 记录此条问题 Q:...
                t = words_list2index(tmp, word2idx, max_len)        # 将 knowledge + B:... + Q:... 词列表转换为 index 序列
            else:
                if line[0] == 'R':              # 对应 i % 6 == 2
                    questions.append(t)         # questions 初始为空，questions <- t(即 knowledge + B:... + Q:... 序列)
                    answers.append(words_list2index(train_texts[i], word2idx, max_len))     # 记录正确答案
                    labels.append(1)                                                        # 标签记为 1
                elif line[0] == 'W':            # 处理错误答案
                    questions.append(t)
                    answers.append(words_list2index(train_texts[i], word2idx, max_len))     # 记录错误答案及标签 0
                    labels.append(0)
                if i % 6 == 5:                  # 每条的 knowledge + B:... + Q:... 序列不同，得清空
                    tmp.clear()
    return questions, answers, labels, question_num


# 逐个获取每一批训练数据的迭代器，会区分每个问题的正确和错误答案，拼接为 (q,a+,a-) 形式，每个一式三份
# (问题列表，答案列表，标签列表，问题 ID 列表，一次训练选择的样本数)
def training_batch_iter(questions, answers, labels, question_num, batch_size):
    """
    :return q + -
    """
    batch_num = int(question_num / batch_size) + 1          # 批数
    for batch in range(batch_num):      # for each batch，对每一批问题
        ret_questions, true_answers, false_answers = [], [], []
        for i in range(batch * batch_size, min((batch + 1) * batch_size, question_num)):    # 每次处理一个 batch_size 或者剩下的 question_num
            # for each question(4 line)
            ix = i * 4
            ret_questions.extend([questions[ix]] * 3)           # 问题列表复制三次
            for j in range(ix, ix + 4):
                if labels[j]:
                    true_answers.extend([answers[j]] * 3)       # 正确答案复制三次
                else:
                    false_answers.append(answers[j])            # 三个错误答案排一起
        yield np.array(ret_questions), np.array(true_answers), np.array(false_answers)      # yield 可简单看为 return，https://blog.csdn.net/mieleizhi0522/article/details/82142856/


# 逐个获取每一批测试数据的迭代器
# (问题列表，答案列表，标签列表，问题 ID 列表，一次训练选择的样本数)
def testing_batch_iter(questions, answers, question_num, batch_size):
    batch_num = int(question_num / batch_size) + 1          # 批数
    questions, answers = np.array(questions), np.array(answers)
    for batch in range(batch_num):                          # 对每一批问题
        start_ix = batch * batch_size * 4
        end_ix = min((batch + 1) * batch_size * 4, len(questions))
        yield questions[start_ix:end_ix], answers[start_ix:end_ix]      # 构成问题和答案集
