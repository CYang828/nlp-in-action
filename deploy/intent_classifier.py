# Streamlit
import streamlit as st

# 数据科学
import pandas as pd

print(f"Pandas: {pd.__version__}")
import numpy as np

print(f"Numpy: {np.__version__}")

# 深度学习
import tensorflow as tf

print(f"Tensorflow: {tf.__version__}")
from tensorflow import keras

print(f"Keras: {keras.__version__}")
import sklearn

print(f"Sklearn: {sklearn.__version__}")

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

import collections
import yaml

# 预处理和Keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import re
import os
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# 读取意图
with open(r"../objects/intents.yml") as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)

# 读取代表性意图
with open(r"../objects/intents_repr.yml") as file:
    intents_repr = yaml.load(file, Loader=yaml.FullLoader)

# 读取训练数据
train = pd.read_pickle("../objects/train.pkl")

print(train.head())
print(f"\n意图:\n{intents}")
print(f"\n代表性意图:\n{intents_repr}")

"""
KERAS 预处理
"""
# 函数定义
def make_tokenizer(docs, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    t = Tokenizer(filters=filters)
    t.fit_on_texts(docs)
    return t

encode_tweets = lambda token, words: token.texts_to_sequences(words)

pad_tweets = lambda encoded_doc, max_length: pad_sequences(
    encoded_doc, maxlen=max_length, padding="post"
)

one_hot = lambda encode: OneHotEncoder(sparse=False).fit_transform(encode)

get_max_token_length = lambda series: len(max(series, key=len))

def preprocess():
    # 1. 创建分词器对象
    token = make_tokenizer(train["Utterance"])

    # 2. 查找词汇量的长度
    vocab_size = len(token.word_index) + 1

    # 3. 查找最大标记长度

    max_token_length = get_max_token_length(train["Utterance"])

    print(f"词汇量大小: {vocab_size} \n最大标记长度: {max_token_length}")

    # 4. 编码文档 - 与 Keras 字典匹配

    encoded_tweets = encode_tweets(token, train["Utterance"])

    # 5. 填充我的文档 - 用标记填充以规范长度

    padded_tweets = pad_tweets(encoded_tweets, max_token_length)
    print("填充后的文档形状:", padded_tweets.shape)
    print("\n编码和填充后的文档预览:\n", padded_tweets)

    # 6. 独热编码以表示目标变量数据（意图）
    # 对其进行排序，以便每次在任何地方初始化此变量时都保持一致
    unique_intents = sorted(list(set(train["Intent"])))

    # 创建另一个分词器
    output_tokenizer = make_tokenizer(
        unique_intents, filters='!"#$%&()*+,-/:;<=>?@[\]^`{|}~'
    )
    encoded_intents = encode_tweets(output_tokenizer, train["Intent"])

    # 为此独热函数重塑编码的文档
    encoded_intents = np.array(encoded_intents).reshape(len(encoded_intents), 1)
    one_hot_intents = one_hot(encoded_intents)
    print(f"\n意图表示预览:\n{one_hot_intents}")

    return (
        max_token_length,
        padded_tweets,
        one_hot_intents,
        token,
        unique_intents,
        vocab_size,
    )

(
    max_token_length,
    padded_tweets,
    one_hot_intents,
    token,
    unique_intents,
    vocab_size,
) = preprocess()
print(unique_intents)

# 7. 分割为训练和测试
X_train, X_val, y_train, y_val = train_test_split(
    padded_tweets,
    one_hot_intents,
    test_size=0.3,
    shuffle=True,
    stratify=one_hot_intents,
)
print(
    f"\n形状检查:\nX_train: {X_train.shape} X_val: {X_val.shape}\ny_train: {y_train.shape} y_val: {y_val.shape}"
)

"""
嵌入矩阵
"""
# 制作自己的特定顺序的嵌入矩阵
d2v_embedding_matrix = pd.read_pickle("../objects/inbound_d2v.pkl")

# 使用 gloVe 词嵌入
embeddings_index = {}
f = open("../models/glove.twitter.27B/glove.twitter.27B.25d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    embeddings_index[word] = coefs
f.close()

print("找到 %s 个词向量。" % len(embeddings_index))

# 初始化所需对象
word_index = token.word_index
EMBEDDING_DIM = 25  # 因为我们使用 25D gloVe 词嵌入

# 获取我的嵌入矩阵
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # 在嵌入索引中找不到的词将是全零的。
        embedding_matrix[i] = embedding_vector

print(embedding_matrix)


def make_model(vocab_size, max_token_length):
    """在此函数中定义神经网络的所有层"""
    # 初始化
    model = Sequential()

    # 添加层 - 对于嵌入层，我确保将我的嵌入矩阵添加到权重参数中
    model.add(
        Embedding(
            vocab_size,
            embedding_matrix.shape[1],
            input_length=max_token_length,
            trainable=False,
            weights=[embedding_matrix],
        )
    )
    model.add(Bidirectional(LSTM(128)))
    # 另一层 LSTM。如果情况不佳，增加密集层大小。
    #    model.add(LSTM(128))
    # 尝试 100
    model.add(
        Dense(600, activation="relu", kernel_regularizer="l2")
    )  # 尝试 50，另一个密集层？这需要一些探索

    # 添加另一个密集层以增加模型复杂性
    model.add(Dense(600, activation="relu", kernel_regularizer="l2"))

    # 仅更新 50% 的节点 - 有助于防止过拟合
    model.add(Dropout(0.5))

    # 最后一层应该是你的意图数量大小！
    # 对于多标签分类使用 sigmoid，否则使用 softmax！
    model.add(Dense(10, activation="softmax"))

    return model


# 实际创建我的模型
model = make_model(vocab_size, max_token_length)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

# 初始化检查点设置以查看进度并保存模型
filename = "../models/intent_classification.h5"

# 学习率调度
# 此函数在前十个时期保持初始学习率
# 并在此后按指数方式减少它。
def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


lr_sched_checkpoint = tf.keras.callbacks.LearningRateScheduler(scheduler)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)


# 这将保存最佳模型
checkpoint = ModelCheckpoint(
    filename, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)

# 你在最后得到的模型是经过 100 个时期的，但那可能不是与验证准确性最相关的权重

# 仅在模型具有最低验证损失时保存权重。提前停止

# 拟合模型
hist = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, lr_sched_checkpoint, early_stopping],
)

"""
可视化
"""
# 可视化训练损失与验证损失（损失是模型的错误程度）
plt.figure(figsize=(10, 7))
plt.plot(hist.history["val_loss"], label="验证损失")
plt.plot(hist.history["loss"], label="训练损失")
plt.title("训练损失 vs 验证损失")
plt.legend()
plt.savefig("plots/intentc_trainval_loss.png")

# 可视化测试准确性与验证准确性
plt.figure(figsize=(10, 7))
plt.plot(hist.history["accuracy"], label="训练准确性")
plt.plot(hist.history["val_accuracy"], label="验证准确性")
plt.title("训练准确性 vs 验证准确性")
plt.legend()
plt.savefig("plots/intentc_trainval_acc.png")

"""
模型步骤
"""

# 我必须重新定义并加载由我的模型检查点保存的模型
from keras.models import load_model

model = load_model("../models/intent_classification.h5")


def infer_intent(text):
    """接受一个话语作为输入，并输出意图概率的字典"""
    # 确保我的文本是字符串
    string_text = re.sub(r"[^ a-z A-Z 0-9]", " ", text)

    # 转换为 Keras 形式
    keras_text = token.texts_to_sequences(string_text)

    # 检查并删除未知词 - [] 表示单词未知
    if [] in keras_text:
        # 过滤
        keras_text = list(filter(None, keras_text))
    keras_text = np.array(keras_text).reshape(1, len(keras_text))
    x = pad_tweets(keras_text, max_token_length)

    # 生成类概率预测
    # 你正在使用过拟合模型进行预测！
    intent_predictions = np.array(model.predict_proba(x)[0])

    # 将概率预测与意图匹配
    pairs = list(zip(unique_intents, intent_predictions))
    dict_pairs = dict(pairs)

    # 输出字典
    output = {
        k: v
        for k, v in sorted(dict_pairs.items(), key=lambda item: item[1], reverse=True)
    }

    return string_text, output


string_text, conf_dict = infer_intent("hi")
print(f"你: {string_text}")
print(f"机器人: \n意图:{conf_dict}")

