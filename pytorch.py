import fileinput

import hanlp
import jieba
import matplotlib.pyplot as plt
import torch
# 张量：可以描述一个复杂或者简单的物理量，是一个多维数组，是一种不随坐标系改变的几何对象
#     一阶张量即向量可以由三个分量表示，二阶张量表示一个几何对象要用两个向量来表示，这两个向量
#     的分量组合有九种，故有九个分量，以此类推
#
# 创建张量：
#
# a = torch.tensor([[1, 2, 4], [4, 4, 6]])
# a = torch.ones([3, 4]，dtype=类型)
# a = torch.zeros([3, 4])
# a = torch.arrange(起, 终, 步长)  按照步长创建序列
# a = torch.linspace(起, 终, 差值)  创建等差数列
# a = torch.rand([3, 4]) 创建三行四列的随机张量，值的范围在0-1
# a = torch.randint(low=,high=,size=[3,4]) 创建三行四列随机整数张量，值的范围在low到high
# a = torch.randn([3, 4])  创建三行四列随机数张量，随机值的分布均值为0，方差为1
#
# 张量属性：
#     .item（） 取出张量中的值
#     .numpy（） 变成numpy数组类型
#     .size（） 形状   .dtype 类型   .dim（） 阶数
#
# tensor的修改：
#     .view(3, 4)  改变形状，以三行四列展示
#     .t()  转置  .transpose(dim1, dim2) 指定两个维度进行转置
# #     .permute(2,0,1) 转置只能一次转两个维度，这个一次能转三个，数字大小代表转前的维度，数字的位置表示转后维度要在的位置
#     .unsqueeze(维度0) 在0维度位置增加一个维度， .squeeze()压缩维度为1维
#     a[:1] 切片，将所有行的第一列切割出来  可以再赋值
#
# CUDA：通用并行计算架构，使得gpu能解决复杂的计算问题
a = torch.cuda.is_available() 判断当前pytorch是否能调用cuda进行计算

tensor数学运算：（适用广播机制）
    a.add(b) a+b  a.sub(b) a-b  a.abs(b) 取绝对值 a.mm() 矩阵运算
简单函数运算：
    .exp 每个元素进行e的指数运算 .sin .cos
原地执行:运算完后的值直接赋值给a，a变化了
    a.add_(b) a+b  a.sub_(b) a-b  a.abs_(b) 取绝对值 a.mm_() 矩阵运算
统计：
    .max .min .mean  .median中位数 .argmax 最大值所在的下标    .max(dim=0) 按照列来查找

pytorch自动求导：
    x = torch.ones(3, 4, requires_grad=True)  或者原地修改  x.requires_grad_(True)
    会追踪接下来关于x的所有操作，方便后面反向传播（在训练的时候参数是要不停改变的，但是在测试的时候就不需要改变了）
    在评估模型的时候可以通过以下代码阻止追踪
    with torch.no_grad():
        评估操作
在requires_grad=True时：tensor.data仅仅是获取tensor中的数据 且要通过tensor.detach().numpy 转换
在requires_grad=False时：tensor.data和tensor等价

梯度计算：
    loss.backward() 根据损失函数，对requires_grad=True的参数去计算梯度

pytorch基础模型组件：
    nn.Module:自定义网络的一个基类，如下：
from torch import nn
    class Lr(nn.Module):
        def __init__(self):
            super(Lr,self).__init__()  # 继承父类的属性和方法
            self.linear = nn.Linear(1, 1) # 声明网络中的组件  nn.Linear为全连接层，参数为输入的数量和输出的数量
        def forward(self, x):  # 实际上是call方法
            out = self.lineara(x)
            return out
model = Lr()  # 实例化模型
predict = model(x)  # 传入数据，计算结果

nn.Sequential:如果说forward函数中比较简单，nn.Sequential会自动完成forward函数的创建
model = nn.Sequential(nn.Linear(2, 64), nn.Linear(64, 1))
x = torch.randn(10, 2)  # 十个样本，两个特征
model(x)

优化器类：已经封装好的用来更新参数的方法
torch.optim.SGD(参数，学习率)
torch.optim.Adam(参数，学习率)
SGD 学习率是定的  Adam 学习率会自动更新
举例：
optimizer = optim.SGD(model.parameters(), lr=1e-3)   实例化 # model.parameters()可获取所有requires_gard=Ture的参数
optimizer.zero_gard() 梯度置为0
loss.backward()  计算梯度
optimizer.step() 更新参数的值

损失函数：
均方误差 nn.MSELoss() 回归问题
交叉熵损失 nn.CrossEntropyLoss 常用于回归问题
如：核心步骤
model = Lr()  # 实例化模型
criterion = nn.MSELoss() # 实例化损失函数
optimizer = optim.SGD(model.parameters(), lr=1e-3)  # 实例化优化器
for i in range(100):
    y_predict = model(x_ture)  # 向前计算预测值
    loss = criterion(y_true, y_predict) # 计算损失结果
    optimizer.zero_gard()  # 当前循环梯度置0
    loss.backward() # 计算梯度
    optimizer.step()  # 更新参数值

model.eval() # 进入评估（预测）模式 固定所有参数不动
model.train()  # 进入训练模式

如果要把代码放到gpu上面跑：加入以下
    device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")
    x,y = x.to(device),y.to(device)
    model = Lr().to(device)
    criterion = nn.MSELoss()
    ....
    predict 和x y要用的时候 predict.cpu().detach().numpy()  x.cpu().data.numpy()


关于pycharm虚拟环境：解释器是关联着一组库的，要到指定的虚拟环境中安装库就是到该环境下的python解释器所在目录下面去pip install

自然语言处理：
    文本预处理：在将语料输入给模型之前要进行一些列的预处理工作，包括文本处理的基本方法、张量表示方法、语料的数据分析、特征处理、数据增强方法
        文本处理基本方法：分词 词性标注 命名实体识别
        文本张量表示方法：one-hot编码 word2vec work embedding
        文本语料的数据分析：标签数量分布 句子长度分布 词频统计与5关键词词云
        文本特征处理：添加n-gram特征 文本长度规范
        数据增强方法：回译数据增强法

        文本处理基本方法：
            分词：将连续的子划分为词语，英文通过空格，中文可通过模块 jieba
                精确模式（生成比较细致的词）：jieba.cut(文本, cut_all=False) 返回生成器   jieba.lcut(文本, cut_all=False) 返回列表
                全模式（把所有符合词语标准的都提取出来）：jieba.lcut(文本, cut_all=True)
                搜索引擎模式（在精确模式的基础上把比较长的词再进行划分）：jieba.lcut_for_search(text)  默认精确模式
                中文繁体分词：和简体一样
                使用用户自定义词典：在一个txt文件中自定义词典 格式 词 频率 词性  jieba.load_userdict("./userdict.txt")

            流行中英文分词工具hanlp：
                中文分词，先引入一个分词器：tokenizer = hanlp.load('CTB6_CINVSEG')
                再tokenizer('text')
                英文分词，tokenizer = hanlp.utils.rules.tokenize_english
                再tokenizer('text')

            命名实体：地名人名机构名等专有名词 命名实体识别，识别这些专有名词
                中文：recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
                英文：recognizer = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_UNCASED_EN)

            词性标注：在分词的基础上，对词进行标注，对文本语言的另一个角度的理解
                jieba标注：import jieba.posseg as pseg     pseg.lcut("我爱北京天安门")  得出的结果是分词后面跟词性
                hanlp标志： tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)  英文就加不同的包
                           tagger(输入是分词结果列表)

        文本张量的表示方法：词表示成向量，向量再合成矩阵，就得到文本张量
            one-hot编码：就是先有一个分好词的列表，然后用0和1组成的变量来表示这些词，词在列表中的位置对应标1，向量的长度等于词的总数
                        from sklearn.externals import joblib   对象的保存与加载
                        from keras.preprocessing.text import Tokenizer  词汇映射器
                        t = Tokenizer(num_words=None, char_leveal=False)
                        t.fit_on_texts(list)  使用映射器拟合现有数据
                        for token in list:
                            zero_list = [0]*len(list)
                            token_index = t.texts_to_sequences([token])[0][0] - 1    sequence序列
                            zero_list[token_index] = 1
                        tokenizer_path = "./Tokensizer"
                        joblib.dump(t, tokenizer_path)

                        调用之前存好的onehot编码：t = joblib.load(tokenizer_path)
                        优势：操作简单，容易理解 劣势：割裂了词与词之间的联系，大语料情况下向量长度过大

            word2vec：将词汇表示成向量的无监督训练方法，通过构建神经网路模型，将网络参数作为词汇的向量表示，包含CBOW和skipgram两种模式
                      将onehot编码变成另外一种更综合更低维度的向量
                    CBOW模式：选取一个长度（窗口）的语料先来作为研究对象，在这个长度中使用上下文词汇表示目标词汇，然后窗口往后推移类推表示
                            过程：a、上下文词汇的onehot编码乘变换矩阵后再相加得到上下文表示矩阵，变换矩阵的形状=onehot的维数*最后生成的矩阵的维数即窗口要移动几次
                                 b、上下文表示矩阵和参数矩阵相乘得到和onehot同维度的结果矩阵，和原来的进行损失计算，更新参数完成一轮模型迭代
                                 c、窗口后移，遍历所有，得出最终的变换矩阵，变换矩阵再和原来的各onehont相乘，得出最终的word2vec表示结果，形状为移动次数*1
                    skipgram模式：用中间词去表示上下文的词，过程和CBOW相似

                    使用fasttext工具实现word2vec的训练和使用：
                        第一步：获取数据
                        第二步：训练词向量
                               import fasttext
                               model = fasttext.train_supervisred('文件夹名/文件名')  再回车就开始对文件中的数据进行训练了
                               model.get_word_vector("词名") 获取指定词汇的词向量
                        第三步：模型超参数的设定
                            model = fasttext.train_supervisred('文件夹名/文件名', "cbow", dim=300, epoch=1, lr=0.1, thred=8)
                            数据  模式：默认skopgram 维度：用几维的向量来表示一个词，默认100 数据循环次数：默5 数据集大的话不用这么多次 学习率：默认0.05 建议0.01-1  线程:默认12 建议和cup核数相同
                        第四步：模型效果检验
                            一种简单方法是查看某词临近的词，通过主观判断这些临近词是否与目标单词相关
                            model.get_nearest_neighbors('sports')  就会返回诸如运动服等词汇及相似度
                        第五步：模型的保存与重加载
                            保存：model.save_model("文件名.bin")
                            加载：model = fasttext.load_model("文件名.bin")
                                 model.get_word_vector("the")

            word embedding:通过一定的方式将词汇映射到指定（更高）维度的空间，广义的word embedding包括所有密集词汇向量的表示方法，如word2vec
                           狭义的word embedding指在神经网络中加入的embedding层，对整个网络进行训练的同时产生的embedding矩阵，这个矩阵就是训练
                           过程中所有输入词汇的向量表示组成的矩阵（将输入进行多次柔和，加入东西糅合综合得到一个更复杂的向量来表示词汇）
                           writer = SummerWriter()   实例化一个摘要写入对象
                           embedded = torch.randn(100, 50)  初始化一个矩阵，假设就是已经得到的词嵌入矩阵
                           meta = list(map(lambda x: x.strip(), fileinput.FileInput("./vocab100.csv")))  导入准备好的中文词汇
                           writer.add_embedding(embedded, metadata=meta)
                           writer.close

        文本数据分析：理解数据，方便选择超参数等  画散点图、饼状图等来分析
            标签数量分布：标记语料中各类数据的占比，比如酒店评价中的好评差评占比
            句子长度分布：看短文本为主还是长文本为主，句子长度的分布情况，对输入张量的尺寸有个参照
            词频统计和关键词词云：
                获取不同的词的数量：
                    chain方法用于扁平化（去除花哨的装饰，让信息本身作为重点凸显出来）列表，from itertools import chain
                    train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data["sentence"]))) set变成集合
                    len(train_vocab)  先分词，再变集合去重，再统计出看有多少不重复的词
                获取高频词词云（以形容词举例）：
                    使用jieba中的词性标注功能
                        import jieba.posseg as pseg
                        def get_a_list(text):
                            ”“”获取形容词列表“”“
                            r = []
                            for g in pseg.lcut(text):
                                if g.flag == "a":   词性属性
                                    r.append(g.word)   词汇属性
                            return r
                    导入绘制词云的工具包
                        from wordcloud import WordCloud
                        def get_word_cloud(keywords_list):
                            wordcloud = WordCloud(font_path="./SimHei.ttf", max_word=100, background_color="white")
                                                  字体包的路径    显示多少个词  背景颜色
                            keywords_string = " ".join(keywords_list)  词云生成器需要传入的是字符串
                            wordcloud.generate(keywords_string)  生成词云
                            “”“绘制图像并显示”“”
                            plt.figure()
                            plt.imshow(wordcloud, interpolation="bilinear")
                            plt.axis("off")
                            plt.show()

        文本特征处理：为语料增加具有普适性的文本特征，即在现有特征的基础上两两或者三三结合生成新的一组特征，以及对加入特征后的语料进行
                    必要处理比如长度规范，可以有效的将重要的文本特征加入训练模型中，增强模型评估指标，常见文本特征处理方法有n-gram  文本长度规范
            n-gram:相邻共现特征，即n个词或者字相邻且共同出现记为一个特征，bi-gram 2 tri-gram 3
                   ngram_range = 2
                   input_list = [1, 3, 2, 1, 5, 3]
                   def create_ngram_set(input_list):
                       return set(zip(*[input_list[i:] for i in range(ngram_range)]))
                   res = create_ngram_set(input_list)
                   print(res)
                   输出效果：{(3,2), (1,3), (2,1), (1,5), (5,3)}

            文本长度规范：模型的输入需要等尺寸大小的矩阵，故在进入模型前需要对每条文本数值映射后的长度进行规范，先根据句子长度分析出覆盖大多数
                        文本的合理长度，对超长文本进行截断，对不足的文本用0进行补齐 减前面 补前面
                        from keras.preprocessing import sequence
                        sequence.pad_sequences(x_train, cutlen)  x_train文本的张量表示，cutlen涵盖%90左右的语料的最短长度

            文本数据增强：
                方法 回译数据增强法：使用google翻译接口，将中文语料翻译成其他语言，再翻译回中文，把得到的新语料加入原语料中即认为对原数据进行了增强
                        优点：操作简单，得到的语料质量高 缺点：可能存在很高的重复率  可通过多转几种语言来解决，建议最多采用三次连续翻译 多了可能效率会低下 失真
                from googletrans import Translator
                translator = Translator()
                translators = translator.translate([语句1，语句2，语句3], dest='ko')  韩文
                ko_res = list(map(lambda x: x.text, translations))  中变韩文 再换成列表

                translators = translator.translate(ko_res, dest='zh-cn')  韩文
                cn_res = list(map(lambda x: x.text, translations))  韩变中 再换成列表

                jieba分词词性表  hanlp词性对照表

    经典序列模型：以下两个都是用来解决文本序列标注问题，如词性标注，命名实体识别
        HMM模型（隐马尔可夫模型）：隐含序列：隐含信息（直接看不到的信息比如文本的词性）的序列，输入为文本，输出为隐含序列
                               隐体现在假设隐藏序列中每个单元的可能性只与上一个单元有关
        CRF模型（条件随机场）：文本序列作为输入，该序列对应的隐含序列作为输出
    
    RNN（循环神经网络）架构：序列输入序列输出，通过网络内部结构有效捕捉序列之间的关系特征，结构：输入进入中间层，得出一个结果，可以直接输出这个结果，
                         下一次循环结合了该结果和新输入；很好的利用了序列之间的关系，广泛应用于具有连续性的输入序列，如人类语言语音识别、文本分类
                         情感分析等；按照输入输出划分有：n入n出（写诗） n入1出（文本分类） 1入n出（看图说话） n入m出（编码器和解码器组成，
                         编码器先根据输入生成一个隐含变量c，再把c作为解码器的输入  也叫seq2seq架构）

        传统RNN：import torch
                import torch.nn as nn
                rnn = nn.RNN(5, 6, 1) 输入张量x特征维度 隐层张量h特征维度 隐含层数量 激活函数选择默认tanh
                input = torch.randn(1, 3, 5)  最后一个参数对应上面的维度
                h0 = torch.randn(1, 3, 6)
                output, hn = rnn(input, h0)
            简单 短序列表现优异，但在解决长序列时，进行反向传播会发生梯度消失或者梯度爆炸（连乘） tanh函数将值压缩到-1-1之间

        LSTM（长短时记忆结构）模型：能够有效捕捉长序列之间的语义联系，缓解梯度消失和梯度爆炸现象 在传统RNN的基础上中间层复杂得多
            结构：遗忘门：过滤掉上一次循环结果的部分 输入门 细胞状态 输出门单向的
                import torch
                import torch.nn as nn
                rnn = nn.LSTM(5, 6, 2) 输入张量x特征维度 隐层张量h特征维度（神经元数量） 隐含层数量 激活函数选择默认tanh 是否选择双向lstm
                input = torch.randn(1, 3, 5)  最后一个参数对应上面的维度
                h0 = torch.randn(2, 3, 6)
                c0 = torch.randn(2, 3, 6)  初始化细胞状态张量
                output, （hn，cn） = rnn(input, (h0,c0))

        GRU（门控学习单元）模型：继承了经典和LSTM的优点,但是不能并行计算   结构：更新门 充值门
                import torch
                import torch.nn as nn
                rnn = nn.GRU(5, 6, 1) 输入张量x特征维度 隐层张量h特征维度 隐含层数量 激活函数选择默认tanh
                input = torch.randn(1, 3, 5)  最后一个参数对应上面的维度
                h0 = torch.randn(1, 3, 6)
                output, hn = rnn(input, h0)

    注意力机制：主要用在seq2seq架构，在解码器端的注意力机制：能根据目标有效的聚焦编码器的输出结果，关注重点信息，改善编码器输出是单一张量无法存储过多信息的情况
              在编码器端的注意力机制：主要解决表征问题，相当于特征提取过程，得到输入的注意力表示，一般自注意
            实现步骤：1、根据注意力计算规则对q k v进行相应计算
                    2、如果第一步采用的是拼接方法则需要将q与第二步的计算结果进行拼接，如果是转置点积，一般是自注意力，q与v相同，则不需要进行与q的拼接
                    3、使用线性层作用在第二步的结果上做一个线性变换，得到最终对q的注意力表示

    transformer：

    fastText工具使用：（用于文本分类和训练词向量）
        文本分类：二分类 单标签多分类 多标签多分类
        使用fasttest进行文本分类过程：
            1、获取数据  2、训练集验证集划分 3、训练模型 4、使用模型进行预测并评估 5、模型调优 6、模型保存与重加载
            import fasttext
            model = fasttext.train_supervisred(input="训练集数据名字",lr=1.0, epoch=25)  训练模型(训练25轮)
            model.predict("任意一个文本")  预测
            返回一个元组（标签，概率）
            模型调优：从原始数据入手（大小写 规范等），增加训练轮数,调整学习率，修改损失计算方式（改一个参数）
                    自动超参数调优model = fasttext.train_supervisred(input="训练集数据名字", autotuneValidation='训练集'，autotuneDuration=600)

    训练词向量：
        词向量：用向量表示文本中的词汇或字符
        fasttext训练词向量步骤：获取数据 训练词向量 模型超参数设定 模型效果检验 模型保存与加载
        model = fasttext.train_supervisred('data/fil9')  传输词汇集文件进行训练
        model.get_word_vector("the")  查看某一词的词向量
        模型超参数设定：无监督训练模式，skipgram或者cbow 词嵌入维度dim 循环次数 学习率 线程数
        检验：model.get_nearest_neighbors('music')
            model.save_model("filename.bin")
            model = fasttext.load_model("filename.bin")

    词向量迁移：使用大型语料库上已经训练完成的词向量模型
    使用fasttest进行词向量模型迁移：
        1、下载词向量模型压缩的bin.gz文件
        2、解压bin.gz文件到bin文件
        3、加载bin文件获取词向量  model = fasttext.load_model("解压好的bin")
        4、利用临近词进行效果检验

    迁移学习：
        预训练模型：比较大型且复杂 常见有BERT  GPT  roBERTTa
        微调:调整预训练模型的部分参数或者一些结构然后在小部分数据集上进行训练，使整个模型更好的适应训练任务
        两种迁移学习：直接用和微调了之后再用
        NLP中的标准数据集：GLUE数据集（包含：cola sst-2 mrpc sts-b qqp mnli snli qnli rte wnli ） 有下载脚本
        NLP中的常用预训练模型：BERT GPT GPT-2 Transformer-xl XLNet XLM RoBERTa DistilBERT T5 XLM-RoBERTA
        加载和使用预训练模型：
            1、确定需要加载的预训练模型并安装依赖包
            2、加载预训练模型的映射器tokenizer
                import torch
                tokenizer = torch.hub.load(source, part, model_name)
                            预训练模型的来源，选定加载模型的哪一部分，加载的预训练模型的名字
            3、加载带或者不带头的预训练模型（带不带任务输出层）
            4、使用模型获得输出结果
