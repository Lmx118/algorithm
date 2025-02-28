import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Cnames=['label','message']
data=pd.read_csv('SMSSpamCollection.txt',sep='\t', header=None, names=Cnames)
data=data.replace({'ham':0,'spam':1})  #替换标签值


# 对文本数据进行特征提取,创建词袋
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])

# 目标变量
y = data['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并拟合朴素贝叶斯模型
model = MultinomialNB()
model.fit(X_train, y_train)

# 在测试集上进行预测
predictions = model.predict(X_test)

print('多项式模型的前一百个预测结果：')
print(predictions[0:100])
# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print('准确率:', accuracy)

# 输出混淆矩阵和分类报告
print('混淆矩阵:')
print(confusion_matrix(y_test, predictions))
print('分类报告:')
print(classification_report(y_test, predictions))

