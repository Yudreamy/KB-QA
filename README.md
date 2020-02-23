# Knowledge-Based Question Answering

基于知识库的中文问答系统.
整体流程如下:
1. 根据Background和Question寻找到最相关的K个Knowledge，`K Knowledge+Background+Question`构成一个`大问题`.
2. `正确选项`分别与该问题中所有`错误选项`组合，构成3个答案组合，分别与`大问题`组合构成3个样例，采用**余弦距离**计算`大问题`与`正确选项`和`错误选项`的相似度.

    正确选项相似度为t_sim, 错误选项相似度为f_sim,损失函数为

        loss = max(0, margin - t_sim + f_sim)


## Model
- 寻找相关Knowledge: LSI
- 训练: biLSTM

## Requirement

- python3, tensorflow
- stop_words, 中文[word2vec](https://pan.baidu.com/s/1miBYRgO)(a2u6)

## Data Format

- knowledge

        地球是宇宙中的一颗行星，有自己的运动规律。
        地球上的许多自然现象都与地球的运动密切相关。
        地球具有适合生命演化和人类发展的条件，因此，它成为人类在宇宙中的唯一家园。
        ...

- train&test

    问题为选择题，每个问题的格式为
    `Background, Question, Right, Wrong, Wrong, Wrong`.

        B:近年来，我国有些农村出现了“有院无人住，有地无人种”的空心化现象。
        Q:“有院无人住，有地无人种”带来
        R:土地资源浪费
        W:农业发展水平提高
        W:城乡协调发展
        W:农村老龄化程度降低

        B:广东省佛山市三水区被称为“中国饮料之都”。除青岛啤酒、伊利等国内著名饮料企业抢先布局外，百威、红牛、可口可乐、杨协成等国际巨头也先后落户于此，作为其在中国布局中的重要一环。
        Q:众多国际饮料企业选址三水的主导区位因素是
        R:市场
        W:技术
        W:劳动力
        W:原料

        B:凡是大气中因悬浮的水汽凝结，能见度低于1千米时，气象学称这种天气现象为雾
        Q:深秋到第二年初春，晴朗的夜晚容易形成雾，这主要是因为
        R:晴朗的夜晚大气逆辐射弱，近地面降温快
        W:晴天大气中的水汽含量多
        W:晴朗的夜晚地面水汽蒸发强
        W:晴天大气中的凝结核物质较少

        ...

## Usage

python3 train.py

该数据集下最佳参数为
- dropout:0.45
- k:0.5


# 推荐阅读（ME）

主线推荐
- [x] [gensim 使用方法](https://blog.csdn.net/u014595019/article/details/52218249)
- [ ] [gensim - num_topics](https://blog.csdn.net/questionfish/article/details/46742403)
- [ ] [gensim - Similarity Queries](https://blog.csdn.net/questionfish/article/details/46746947)
- [ ] [使用预训练的 word embedding](https://blog.csdn.net/weixin_42101286/article/details/90296819)
- [ ] [LSTM 代码](https://blog.csdn.net/Wzz_Liu/article/details/85038746)

支线推荐
- [x] [batch_size](https://blog.csdn.net/qq_34886403/article/details/82558399)
- [x] [epoch](http://www.360doc.com/content/18/0417/18/47852988_746429309.shtml)
- [x] [checkpoint](https://www.jianshu.com/p/adfa8aa2cbdf)
- [ ] [jieba 分词使用方法](https://blog.csdn.net/laobai1015/article/details/80420016)
- [ ] [Latent Semantic Indexing](https://www.jianshu.com/p/28f2bc62a75b)
- [x] ["//" 地板除](https://blog.csdn.net/qq_29566629/article/details/95374971)
- [ ] [python yield](https://blog.csdn.net/mieleizhi0522/article/details/82142856/)
- [ ] [tf.ConfigProto 和 tf.GPUOptions 用法总结](https://blog.csdn.net/lty_sky/article/details/91491302)
- [x] [有关 global_step](https://blog.csdn.net/leviopku/article/details/78508951)
- [x] [tf.nn.embedding_lookup()](https://blog.csdn.net/laolu1573/article/details/77170407)
- [ ] [tf.reduce_sum()](https://www.jianshu.com/p/30b40b504bae)
- [ ] [tf.nn.max_pool](https://blog.csdn.net/mao_xiao_feng/article/details/53453926)
- [ ] [Hinge Loss](https://blog.csdn.net/ZJRN1027/article/details/80170966)
- [ ] [tf.summary](https://www.cnblogs.com/lyc-seu/p/8647792.html)
- [x] [tf.flags](https://blog.csdn.net/weixin_33896726/article/details/90134617)


> p.s. 主线必读，支线随意

# 说明（Me）
此文件夹下的内容是我在跑了 [DouYishun/KB-QA](https://github.com/DouYishun/KB-QA) 模型然后通读下所有代码的产物，保留了终端输出记录文件 `no0221.txt`、程序模型（res）以及生成的summary(貌似文件太大 >100MB GitHub 无法上传)，这个是在服务器 GPU 上完整跑完了的，另外一个 `0214-no.log` 是在自己电脑上跑生成的记录文件（由于内存不够只跑了一部分）。另外，`KB-QA-master.zip` 是我把向量文件及所有程序打包，解压后放在 D 盘，环境配好后，直接 `python train.py` 即可运行。

TensorBoard 是一个将 summary 可视化的工具，可以用来展示网络图、loss 变化、张量等，参见下图，[使用参考](https://blog.csdn.net/duanlianvip/article/details/98498826)。

![](https://github.com/Yudreamy/KB-QA/blob/master/picture/1.PNG)

![](https://github.com/Yudreamy/KB-QA/blob/master/picture/2.PNG)

![](https://github.com/Yudreamy/KB-QA/blob/master/summary/file.PNG)
