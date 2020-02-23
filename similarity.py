import jieba.posseg         # jieba 分词，词性标注
import os
import codecs
import pickle               # 将程序运行中的对象保存为文件
from gensim import corpora, models, similarities    # gensim 是一个 python 自然语言处理库，能够将文档根据 TF-IDF，LDA，LSI 等模型转化为向量模式，https://blog.csdn.net/u014595019/article/details/52218249
from data_util import tokenizer         # 使用 data_util.py 中自定义的 tokenizer 分词


# 将 knowledge 和 train 文件分词后形成词典，并存储
def generate_dic_and_corpus(knowledge_file, file_name, stop_words):
    knowledge_texts = tokenizer(knowledge_file, stop_words)         # 分词
    train_texts = tokenizer(file_name, stop_words)

    # 在将文档分词后，使用 dictionary = corpora.Dictionary(texts) 生成词典，并可以使用 save 函数将词典持久化
    dictionary = corpora.Dictionary(knowledge_texts + train_texts)  # dictionary of knowledge and train data
    dictionary.save(os.path.join('D:/KB-QA-master/tmp/dictionary.dict'))

    # 生成词典以后 corpus = [dictionary.doc2bow(text) for text in texts] 转化为向量形式，见 gensim 使用方法
    corpus = [dictionary.doc2bow(text) for text in knowledge_texts]  # corpus of knowledge
    # corpora.MmCorpus.serialize 将 corpus 持久化到磁盘中；corpus = corpora.MmCorpus('tmp.mm')，从磁盘中读取
    corpora.MmCorpus.serialize('D:/KB-QA-master/tmp/knowledge_corpus.mm', corpus)



# 将 file_name(即 train/test 文件) 逐条处理，将每条的 B:...，Q:... 结合为 vec_lsi
# 为每个 vec_lsi 在 corpus 语料库中找 k 个最相关的 knowledge 条目，并存入文件
def topk_sim_ix(file_name, stop_words, k):
    sim_path = "D:/KB-QA-master/tmp/" + file_name[5:-4]     # 参考 data/train.txt -> train 格式 0 1 2 3 4 /5.../ 去掉后 4 位
    if os.path.exists(sim_path):            # 如果之前运行过，就存在该文件，直接导入
        with open(sim_path, "rb") as f:
            sim_ixs = pickle.load(f)
        return sim_ixs

    # load dictionary and corpus    导入 generate_dic_and_corpus 函数生成的词典及向量化形式文件
    dictionary = corpora.Dictionary.load("D:/KB-QA-master/tmp/dictionary.dict")  # dictionary of knowledge and train data
    corpus = corpora.MmCorpus("D:/KB-QA-master/tmp/knowledge_corpus.mm")  # corpus of knowledge

    # build Latent Semantic Indexing model
    # gensim->models 可以对 corpus 进行进一步的处理，比如使用 tf-idf 模型，lsi 模型，lda 模型等，非常强大 
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=10)  # initialize an LSI transformation, num_topics 选项控制语料转换为一个潜在 n（num_topics）维 topic 空间，https://blog.csdn.net/questionfish/article/details/46742403

    # similarity，负责计算文档之间的相似度
    index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
    sim_ixs = []  # topk related knowledge index of each question，对每个问题计算 k 个最相关的 knowledge
    with open(file_name,encoding='utf8') as f:
        tmp = []  # background and question
        for i, line in enumerate(f):
            if i % 6 == 0:              # 对应 B:...
                tmp.extend([token for token, _ in jieba.posseg.cut(line.rstrip()) if token not in stop_words])      # 同分词处理
            if i % 6 == 1:              # 对应 Q:...
                tmp.extend([token for token, _ in jieba.posseg.cut(line.rstrip()) if token not in stop_words])
                vec_lsi = lsi[dictionary.doc2bow(tmp)]  # convert the query to LSI space，分词生成词典后转换为向量形式
                # 获得我们的查询文档相对于之前生成的 corpus 语料库的相似度，https://blog.csdn.net/questionfish/article/details/46746947
                sim_ix = index[vec_lsi]  # perform a similarity query against the corpus
                sim_ix = [i for i, j in sorted(enumerate(sim_ix), key=lambda item: -item[1])[:k]]  # topk index，取topk
                sim_ixs.append(sim_ix)
                tmp.clear()
    with open(sim_path, "wb") as f:         # 写入 f
        pickle.dump(sim_ixs, f)
    return sim_ixs


# module test，测试部分，similarity.py 可直接运行
# 运行结果输出 10190，符合 train 文件 61140（=10190*6）
if __name__ == '__main__':
    stop_words_ = codecs.open("data/stop_words.txt", 'r', encoding='utf8').readlines()
    stop_words_ = [w.strip() for w in stop_words_]
    generate_dic_and_corpus("data/knowledge.txt", "data/train.txt", stop_words_)
    res = topk_sim_ix("data/train.txt", stop_words_, 5)
    print(len(res))

