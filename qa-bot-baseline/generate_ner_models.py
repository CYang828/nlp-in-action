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

# 命名实体识别
import spacy

print(f"spaCy: {spacy.__version__}")
from spacy import displacy
import random
from spacy.matcher import PhraseMatcher
import plac
from pathlib import Path

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

# Cool progress bars
from tqdm import tqdm_notebook as tqdm

tqdm().pandas()  # 启用执行进度跟踪

import collections
import yaml
import pickle

# 读取意图
with open(r"../objects/intents.yml") as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)

# 读取代表意图
# with open(r'../objects/intents_repr.yml') as file:
#     intents_repr = yaml.load(file, Loader=yaml.FullLoader)

# Cool progress bars
from tqdm import tqdm_notebook as tqdm


# 加载spaCy
nlp = spacy.load("en_core_web_sm")

# 读取训练数据
train = pd.read_pickle("../objects/train.pkl")

print(train.head())
print(f"\nintents:\n{intents}")

# 读取处理后的数据
processed = pd.read_pickle("../objects/processed.pkl")

# 看起来我必须制作自己的训练数据

entities = {
    "hardware": [
        "macbook pro",
        "iphone",
        "iphones",
        "mac",
        "ipad",
        "watch",
        "TV",
        "airpods",
    ],
    "apps": [
        "app store",
        "garageband",
        "books",
        "calendar",
        "podcasts",
        "notes",
        "icloud",
        "music",
        "messages",
        "facetime",
        "catalina",
        "maverick",
    ],
}

# 将其存储到YAML文件
with open("../objects/entities.yml", "w") as outfile:
    yaml.dump(entities, outfile, default_flow_style=False)

# 读取数据

hardware_train = pd.read_pickle("../objects/hardware_train.pkl")
app_train = pd.read_pickle("../objects/app_train.pkl")

# 预览
print(hardware_train[:5])
print(app_train[:5])

"""
使用SGD训练识别器
"""


# 现在我们训练识别器。
def train_spacy(train_data, iterations):
    nlp = spacy.blank("en")  # 创建空的Language类
    # 创建内置管道组件并将其添加到管道
    # nlp.create_pipe适用于已在spaCy中注册的内置组件
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe("ner", last=True)

    # 添加标签
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # 在训练期间禁用除'ner'之外的所有管道
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # 仅训练NER
        optimizer = nlp.begin_training()

        train_loss = []

        # 多次遍历训练数据
        for itn in range(iterations):
            print("开始迭代" + str(itn))

            # 打乱训练数据
            random.shuffle(train_data)

            # 迭代级别指标
            losses = {}
            misalligned_count = 0

            # 遍历每个Tweet
            for text, annotations in train_data:
                try:
                    nlp.update(
                        [text],  # 文本批次
                        [annotations],  # 注释批次
                        drop=0.2,  # 丢失率 - 使记忆数据更困难
                        sgd=optimizer,  # 用于更新权重的可调用函数
                        losses=losses,
                    )
                except ValueError as e:
                    misalligned_count += 1
                    # 如果到这里，那意味着有不匹配的实体
                    print(f"忽略不匹配的实体...\n{(text,annotations)}")
                    pass

            # 如果要跟踪不匹配的计数，请启用此选项
            #             print(f'-- misalligned_count (iteration {itn}): {misalligned_count}')
            # 记录损失
            train_loss.append(losses.get("ner"))
            print(f"损失（迭代{itn}）：{losses}")

        # 可视化损失
        plt.figure(figsize=(10, 6))
        plt.plot([*range(len(train_loss))], train_loss, color="magenta")
        plt.title("每次迭代的损失")
        plt.xlabel("迭代次数")
        plt.ylabel("损失")
        plt.show()

    return nlp


# 错误率正在上升，达到了我们当前路径的最小值
# 我们选择20次迭代，但有一点是，如果你做太多次，它会忘记现在知道的东西

# 训练1
hardware_nlp = train_spacy(hardware_train, 20)

# 将我们训练好的模型保存到新目录中
hardware_nlp.to_disk("../models/hardware_big_nlp")

# 训练2
app_nlp = train_spacy(app_train, 10)

# 将我们训练好的模型保存到新目录中
app_nlp.to_disk("../models/app_big_nlp")

# 序列化
pickle.dump(hardware_nlp, open("../models/hardware_big_nlp.pkl", "wb"))
pickle.dump(app_nlp, open("../models/app_big_nlp.pkl", "wb"))
