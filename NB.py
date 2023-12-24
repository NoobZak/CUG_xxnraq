# 正则表达式运算
import re
# 中文文本分词库
import jieba
# 用于数学运算
import numpy as np
# counter用来计数器
from collections import Counter
# tqdm库用来画进度条
from tqdm import tqdm
# logging库用来组织jieba库打印日志
import logging
# 设置jieba库的日志级别
jieba.setLogLevel(logging.INFO)

# 读取停止词
stopword = open('./stopWord.txt', encoding='utf-8').read().split('\n')
# 包含词语的文件数
HamWords = Counter()
SpamWords = Counter()


def countwords(label, path):
    wordtimes = Counter()
    with open(path, 'rb') as f:
        content = f.read()
        content = content.decode('gbk', 'ignore')
    # 用utf-8格式读取邮件
    text = content.encode('utf-8', 'ignore').decode('utf-8')
    # 使用正则表达式过滤掉所有非中文字符，替换为空字符。为了去除非关键因素
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用jieba库进行分词
    text = jieba.cut(text, cut_all=False)
    # 如果word不是停止词，则给该词汇频率加1
    for word in text:
        if len(word) > 1 and word not in stopword:
            wordtimes[word] += 1
    # 将分词组成新的数组
    for word in wordtimes:
        if wordtimes[word] >= 1:
            if label == 'ham':
                HamWords[word] += 1
            if label == 'spam':
                SpamWords[word] +=1
    f.close()

# 读取邮件函数
def reademails():
    # --------------------文件常规处理-----------------------
    # 正常邮件标志
    HamNum = 0
    # 垃圾邮件标志
    SpamNum = 0
    index_path = './trec06c/full/index'
    with open(index_path) as f:
        lines = f.readlines()
    # 读取前80%的数据作为训练集
    lines = lines[:int(0.8 * len(lines))]
    # 创建进度条pbar total=len(lines) 进度条长度为lines
    # --------------------文件常规处理-----------------------
    with tqdm(total=len(lines)) as pbar:
        pbar.set_description("计算词语频率...")
        for line in lines:
            # index中关于邮件描述的关键字存入label
            label = line.split(' ')[0]
            # ./trec06c+split后第二段（地址）内容，从第三个字符开始取，且将末尾的换行符换成空格
            path = './trec06c' + line.split(' ')[1].replace('\n', '')[2:]
            if label == 'ham':
                HamNum += 1
            if label == 'spam':
                SpamNum += 1
            countwords(label, path)
            pbar.update()
    return HamNum, SpamNum

# 贝叶斯分类器模型训练
def train(HamNum, SpamNum):
    # 计算垃圾邮件和正常邮件的先验概率
    PHam = HamNum / (HamNum + SpamNum)
    PSpam = 1 - PHam
    # 生成每个词汇的概率字典
    with tqdm(total=len(HamWords) + len(SpamWords)) as pbar:
        pbar.set_description("训练模型...")
        for word in HamWords:
            # 通过对数运算，避免了小概率的相乘导致的数值下溢问题
            HamWords[word] = np.log(HamWords[word]) - np.log(HamNum)
            pbar.update()
        for word in SpamWords:
            SpamWords[word] = np.log(SpamWords[word]) - np.log(SpamNum)
            pbar.update()
    return PHam, PSpam


def test(path, PHam, PSpam):
    # 判断为Ham和Spam的概率
    PH = 1
    PS = 1
    # --------------------邮件常规处理-----------------------
    with open(path, 'rb') as f:
        content = f.read()
        content = content.decode('gbk', 'ignore')
    # 用utf-8格式读取邮件
    text = content.encode('utf-8', 'ignore').decode('utf-8')
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用jieba库进行分词
    text = jieba.lcut(text, cut_all=False)
    # --------------------邮件常规处理-----------------------
    PH = np.log(PH)
    PS = np.log(PS)
    for word in text:
        if word not in stopword:
            if word in HamWords and word in SpamWords:
                # 朴素贝叶斯分类器假设：各个单词之间是条件独立的
                PH = PH + HamWords[word]
                PS = PS + SpamWords[word]
    PH = PH + np.log(PHam)
    PS = PS + np.log(PSpam)
    # 如果判断为Ham则返回True
    if PH > PS:
        return 'ham'
    else:
        return 'spam'


# 检测模型准确性
# 准确率（Accuracy）： 它是分类器正确分类的样本数量与总样本数量的比例。准确率越高，表示分类器在所有样本中分类正确的比例越高。其计算公式为：
# Accuracy = TP + TN / (TP + TN + FP + FN)
# 查准率（Precision）： 它是分类器预测为正类别的样本中，实际为正类别的样本数量占比。查准率高表示分类器识别为正类别的样本中真正为正类别的比例高。计算公式为：
# Precision = TP / (TP + FP)
# 召回率（Recall）： 它是实际为正类别的样本中，被分类器正确预测为正类别的样本数量占比。召回率高表示分类器对正类别的样本识别能力强。计算公式为：
# Recall = TP /(TP + FN)
def detection(PHam, PSpam):
    # True Positive :正确识别为正常邮件
    TP = 0
    # True Negative ：正确识别为垃圾邮件
    TN = 0
    # False Positive ：正常邮件错误识别为垃圾邮件 弃真错误
    FP = 0
    # False Negative ：垃圾邮件错误识别为正常邮件 纳伪错误
    FN = 0
    # --------------------邮件常规处理-----------------------
    index_path = './trec06c/full/index'
    with open(index_path) as f:
        lines = f.readlines()
    # 读取后20%的数据作为测试集
    lines = lines[int(0.8 * len(lines)):]
    # --------------------邮件常规处理-----------------------
    with tqdm(total=len(lines)) as pbar:
        pbar.set_description("测试模型性能...")
        for line in lines:
            label = line.split(' ')[0]
            path = './trec06c' + line.split(' ')[1].replace('\n', '')[2:]
            # test检测模型的准确度
            result = test(path, PHam, PSpam)
            # 将结果与test结果对比
            if result == 'ham' and label == 'ham':
                TP += 1
            if result == 'spam' and label == 'spam':
                TN += 1
            if result == 'ham' and label == 'spam':
                FP += 1
            if result == 'spam' and label == 'ham':
                FN += 1
            pbar.update()
    # 准确率
    accuracy = (TP + TN) / len(lines) * 100
    # 查准率
    precision = TP / (TP + FP) * 100
    # 召回率
    recall = TP / (TP + FN) * 100
    print('测试已完成，数据如下：')
    print('模型计算准确率:{}%'.format(accuracy))
    print('模型计算查准率:{}%'.format(precision))
    print('模型计算召回率:{}%'.format(recall))


if __name__ == '__main__':
    # 读取邮件
    HamNum, SpamNum = reademails()
    # 训练模型
    PHam, PSpam = train(HamNum, SpamNum)
    # 模型评估
    detection(PHam, PSpam)
    with open('./HamWords.txt', 'w') as f1:
        f1.write(str(HamWords))
    with open('./SpamWords.txt', 'w') as f2:
        f2.write(str(SpamWords))
