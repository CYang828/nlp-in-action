import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 加载意图分类模型
from keras.models import load_model

import pandas as pd
import numpy as np
import yaml

train = pd.read_pickle("../objects/train.pkl")
print(f"训练数据: {train.head()}")

model = load_model("../models/intent_classification_b.h5")

# 我使用Keras的Tokenizer API - 我遵循的有用链接: https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
# 训练测试分离
# 分割为训练和测试集
# 为了处理类别不平衡和可重现性，使用stratify和random state - 7是我的幸运数字
X_train, X_val, y_train, y_val = train_test_split(
    train["Utterance"],
    train["Intent"],
    test_size=0.3,
    shuffle=True,
    stratify=train["Intent"],
    random_state=7,
)


# 编码目标变量
le = LabelEncoder()
le.fit(y_train)

# 注意: 由于我们使用了嵌入矩阵，我们使用Tokenizer API对数据进行整数编码 - https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
t = Tokenizer()
t.fit_on_texts(X_train)

# 将文档填充到指定的最大长度
max_length = len(max(X_train, key=len))


def convert_to_padded(tokenizer, docs):
    """接受Keras API Tokenizer和文档，返回它们的填充版本"""
    ## 使用API的属性
    # 嵌入
    embedded = t.texts_to_sequences(docs)
    # 填充
    padded = pad_sequences(embedded, maxlen=max_length, padding="post")
    return padded


padded_X_train = convert_to_padded(tokenizer=t, docs=X_train)
padded_X_val = convert_to_padded(tokenizer=t, docs=X_val)


def infer_intent(user_input):
    """创建一个接收用户输入并输出预测字典的函数"""
    assert isinstance(user_input, str), "用户输入必须是字符串!"
    keras_input = [user_input]
    print(user_input)

    # 转换为Keras形式
    padded_text = convert_to_padded(t, keras_input)
    x = padded_text[0]

    # 每个文档的预测
    probs = model.predict(padded_text)
    #     print('概率数组形状', probs.shape)

    # 从标签编码器获取类别
    classes = le.classes_

    # 获取预测字典并排序
    predictions = dict(zip(classes, probs[0]))
    sorted_predictions = {
        k: v
        for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    }

    # 保存意图分类
    # 将其存储到YAML文件中
    with open("../objects/sorted_predictions.yml", "w") as outfile:
        yaml.dump(sorted_predictions, outfile, default_flow_style=False)

    return user_input, sorted_predictions


# 测试结果
# print(infer_intent("update is not working"))

