# -*- coding:utf-8 -*-

import os
import pickle
import numpy as np
from sklearn import svm


class CLFModel(object):
    def __init__(self, model_save_path):
        super(CLFModel, self).__init__()
        self.model_save_path = model_save_path
        self.id2label = pickle.load(open(os.path.join(self.model_save_path,'id2label.pkl'),'rb'))
        self.vec = pickle.load(open(os.path.join(self.model_save_path,'vec.pkl'),'rb'))
        self.LR_clf = pickle.load(open(os.path.join(self.model_save_path,'LR.pkl'),'rb'))
        self.gbdt_clf = pickle.load(open(os.path.join(self.model_save_path,'gbdt.pkl'),'rb'))

    """
    predict函数
    文本预处理：首先将输入的文本转换为小写，并按照字符逐个划分为一个列表，然后用空格连接起来。这样做的目的可能是将文本转换为空格分隔的字符序列。

    特征提取：利用预先训练好的self.vec（可能是文本特征提取器，如TF-IDF向量化器）对处理后的文本进行向量化。

    模型预测：使用两个模型（self.LR_clf和self.gbdt_clf）分别对向量化后的文本进行预测，并获取预测的概率。

    结果合并：将两个模型的预测概率取平均，然后找出概率最高的类别作为最终的预测结果，并通过self.id2label将预测结果映射为具体的类别标签返回。
    """
    def predict(self,text):
        text = ' '.join(list(text.lower()))  #lower()将字符小写，list组成字符列表，' 'join加空格
        text = self.vec.transform([text])  #将文本转化为向量形式，vec是特征提取器。transformer是将文本转化为向量
        proba1 = self.LR_clf.predict_proba(text)  #使用两个模型（self.LR_clf和self.gbdt_clf）分别对向量化后的文本进行预测，并获取预测的概率。
        proba2 = self.gbdt_clf.predict_proba(text)
        label = np.argmax((proba1+proba2)/2, axis=1)  #将两个模型的预测概率取平均，然后找出概率最高的类别作为最终的预测结果，并通过self.id2label将预测结果映射为具体的类别标签返回。
        return self.id2label.get(label[0])

if __name__ == '__main__':
    model = CLFModel('./model_file/')

    text='你是谁'
    label = model.predict(text)
    print(label)