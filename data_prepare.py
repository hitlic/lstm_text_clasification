# coding: utf-8
"""
@author Liuchen
2018
"""
import re
import pandas as pd
import logging
logger = logging.getLogger('main.data_prepare')


def load_from_csv(csv_file, delimiter=',', lang='EN'):
    """
    从csv文件（默认为逗号分隔）中加载数据
    :param csv_file: csv文件
    :param delimiter: csv文件分隔符
    :param lang: 文本语言，EN为英文文本
    :return: tuple, (x list, y list)
    """
    logger.info("loading data from csv file ... ")
    data = pd.read_csv(csv_file, encoding="utf-8", delimiter=delimiter)

    data = data.drop(data[data.sentiment == 3].index)  # 删除正负情感冲突的微博  ---- 仅中文数据
    data = data.drop(data[data.sentiment == 2].index)  # 删除情绪为surprise的微博  ---- 仅中文数据
    # data.loc[data.sentiment == 2, 'sentiment'] = -1  # 将surprise作为负面情感 ---- 仅中文数据

    texts = data["content"].values
    texts = [preprocess_text(text, lang=lang) for text in texts]
    labels = data["sentiment"].values
    return texts, labels


def load_from_class_files(files):
    """
    从多个类别文件中加载数据，每个文件一个类别，文件数量即类别数量
    :param files: 类别文件列表
    :return: tuple, (x list, y list)
    """
    logger.info("loading data from many text files ...")
    texts = []
    labels = []
    for fid, file in enumerate(files):
        f = open(file, encoding="utf-8")
        file_data = f.readlines()
        texts.extend(file_data)
        f.close()
        labels.extend([fid]*len(file_data))
    return texts, labels


def preprocess_text(text, lang="EN"):
    """
    股票评论文本数据处理
    :param text: String. 文本数据
    :return: List of Strings.  处理后的文本
    """
    if lang == 'EN':  # 处理英文数据
        return clean_englisth(text)
    else:  # 处理中文数据
        return clean_chinese(text)

    return text


def clean_chinese(text):
    """
    中文数据清理，暂无
    """
    return text


def clean_englisth(text):
    """
    英文数据清理
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", text)  # 去除特殊字符
    text = re.sub(r"\'s", " \'s", text)  # 's 替换为 空格+'s
    text = re.sub(r"\'ve", " \'ve", text)  # 'e 替换为 空格+'ve
    text = re.sub(r"n\'t", " n't", text)  # n't 替换为 空格+n't
    text = re.sub(r"\'re", " \'re", text)  # 're 替换为 空格+'re
    text = re.sub(r"\'d", " \'d", text)  # 'd 替换为 空格+'d
    text = re.sub(r"\'ll", " \'ll", text)  # 'll 替换为 空格+'ll
    text = re.sub(r",", " , ", text)  # 标点替换为 空格+标点+空格
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)  # 连续2个或多个空白字符变为一个
    return text.strip().lower()


if __name__ == "__main__":
    x, y = load_from_class_files(['data/pos.txt', 'data/neg.txt'])
    for i in range(10000):
        print(x[i], y[i])
