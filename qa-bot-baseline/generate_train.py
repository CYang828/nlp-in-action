import streamlit as st

import pandas as pd
import numpy as np

# 词嵌入
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import builtins

# 文本
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

# 通过序列化存储为对象
from tempfile import mkdtemp
import pickle
import joblib

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

# 目录
import os
import yaml
import collections
import math

## 加载对象
processed_inbound = pd.read_pickle("../objects/processed_inbound_extra.pkl")
processed = pd.read_pickle("../objects/processed.pkl")

# 重新读取意图
with open(r"../objects/intents.yml") as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)


if __name__ == "__main":
    main()


def main():

    st.title("训练数据生成工具")

    """制作我的理想数据集 - 生成 N 条与这条人工推文相似的推文
    然后将其连接到当前入站数据中，以便包含在 doc2vec 训练中
    """

    # 版本 2 - 我意识到关键词可能能胜任，并且添加更多词语以增强关联性，因为这是 doc2vec
    ideal = {
        "battery": "电池 功率",
        "forgot_password": "密码 账户 登录",
        "payment": "信用卡 付款 支付",
        "update": "更新 升级",
        "info": "信息 详情",
        # "lost_replace": "替换 丢失 消失 交易",
        "location": "最近 苹果 位置 商店",
    }

    def add_extra(current_tokenized_data, extra_tweets):
        """向当前标记数据添加额外推文"""

        # 将这些额外推文存储在列表中以连接到入站数据
        extra_tweets = pd.Series(extra_tweets)

        # 转换为字符串形式
        print("转换为字符串...")
        string_processed_data = current_tokenized_data.apply(" ".join)

        # 将其添加到数据中，更新 processed_inbound
        string_processed_data = pd.concat([string_processed_data, extra_tweets], axis=0)

        # 我们需要一个标记化版本
        tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        #     print('标记化...')
        #     string_processed_data.apply(tknzr.tokenize)
        return string_processed_data

    # 获取加长的数据
    processed_inbound_extra = add_extra(
        processed["Processed Inbound"], list(ideal.values())
    )

    # 将更新后的 processed_inbound 存储为序列化文件
    processed_inbound_extra.to_pickle("../objects/processed_inbound_extra.pkl")
    st.subheader("更新后的入站数据")
    st.dataframe(processed_inbound_extra)
    st.text(
        "如您所见，我追加了我想要在此数据框中找到相似性的文档，\
    这是在我对数据进行 doc2vec 向量化之前需要做的事情。\
    这是因为 doc2vec 模型相似性函数只能在已存在于向量化数据中的推文之间找到相似性。"
    )

    @st.cache
    def train_doc2vec(string_data, max_epochs, vec_size, alpha):
        # 使用 ID 标记每个数据，并使用最节省内存的方式之一，即只使用其 ID
        tagged_data = [
            TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
            for i, _d in enumerate(string_data)
        ]

        # 实例化我的模型
        model = Doc2Vec(
            size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1
        )

        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            print("迭代 {0}".format(epoch))
            model.train(
                tagged_data, total_examples=model.corpus_count, epochs=model.iter
            )
            # 减小学习率
            model.alpha -= 0.0002
            # 固定学习率，无衰减
            model.min_alpha = model.alpha

        # 保存模型
        model.save("../models/d2v.model")
        print("模型已保存")

    if st.button("训练 doc2vec"):
        train_doc2vec(processed_inbound_extra, max_epochs=100, vec_size=20, alpha=0.025)

    # 加载我的模型
    model = Doc2Vec.load("../models/d2v.model")

    # 将我的数据存储为列表 - 这是我将要对其进行聚类的数据
    inbound_d2v = np.array(
        [model.docvecs[i] for i in range(processed_inbound_extra.shape[0])]
    )

    if st.button("保存向量化的 doc2vec"):
        # 保存
        path = "../objects/inbound_d2v.pkl"
        with open(path, "wb") as f:
            pickle.dump(inbound_d2v, f)
        st.text(f"已保存至 {path}")

    st.subheader("Doc2Vec 向量化数据")
    st.dataframe(inbound_d2v)
    st.text(f"形状: {inbound_d2v.shape}")

    """
    查找理想推文的标签
    """
    # 版本 2
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

    ## 仅标记化 ideal 值的所有值，以便能够输入到匹配函数中
    # intents_repr = dict(zip(ideal.keys(), [tknzr.tokenize(v) for v in ideal.values()]))
    # Pythonic 方式
    intents_repr = {k: tknzr.tokenize(v) for k, v in ideal.items()}
    print(intents_repr)

    # 将 intents_repr 保存到 YAML
    with open("../objects/intents_repr.yml", "w") as outfile:
        yaml.dump(intents_repr, outfile, default_flow_style=False)

    # 字典的标签
    tags = []

    tokenized_processed_inbound = processed_inbound.apply(tknzr.tokenize)
    # 查找特定推文的索引位置

    def report_index_loc(tweet, intent_name):
        """接收推文以查找其索引，并返回该推文索引的报告以及代表性推文的外观"""
        try:
            tweets = []
            for i, j in enumerate(tokenized_processed_inbound):
                if j == tweet:
                    tweets.append((i, True))
                else:
                    tweets.append((i, False))
            index = []
            for i in tweets:
                if i[1] == True:
                    index.append(i[0])

            preview = processed_inbound.iloc[index]

            # 将索引附加到字典中
            tags.append(str(index[0]))
        except IndexError:
            print("索引不在列表中，请继续")
            return

        return intent_name, str(index[0]), preview

    # 报告并使用该函数存储索引
    st.text("标记的索引以供查找")
    for j, i in intents_repr.items():
        try:
            st.text("\n{} \n索引: {}\n预览: {}".format(*report_index_loc(i, j)))
        except Exception as e:
            st.text("索引结束")

    # 从 2 个列表创建新字典的 Pythonic 方式
    intents_tags = dict(zip(intents_repr.keys(), tags))

    st.header("意图标签字典")
    st.write(intents_tags)

    """
    实际生成我的训练数据
    """

    ## 获取与第 0 条推文相似的前 n 条推文
    # 这将返回一个元组列表 (i,j)，其中 i 是索引，j 是与标记文档索引的余弦相似度

    # 将所有意图存储在此数据框中
    train = pd.DataFrame()
    # intent_indexes = {}

    # 1. 基于相似性添加意图内容
    def generate_intent(target, itag):
        similar_doc = model.docvecs.most_similar(itag, topn=target)
        # 仅获取索引
        indexes = [int(i[0]) for i in similar_doc]
        #     intent_indexes[intent_name] = indexes
        # 实际查看与第 0 条推文相似的前 1000 条推文，似乎是关于更新的
        # 仅添加值，不添加索引
        # 标记化输出
        return [
            word_tokenize(tweet)
            for tweet in list(processed_inbound.iloc[indexes].values)
        ]

    # 更新训练数据
    for intent_name, itag in intents_tags.items():
        train[intent_name] = generate_intent(1000, itag)

    # 2. 手动添加的意图
    # 这些是剩余的意图
    manually_added_intents = {
        "speak_representative": [
            ["talk", "human", "please"],
            ["let", "me", "talk", "to", "apple", "support"],
            ["can", "i", "speak", "agent", "person"],
        ],
        "greeting": [
            ["hi"],
            ["hello"],
            ["whats", "up"],
            ["good", "morning"],
            ["good", "evening"],
            ["good", "night"],
        ],
        "goodbye": [["goodbye"], ["bye"], ["thank"], ["thanks"], ["done"]],
        "challenge_robot": [
            ["robot", "human"],
            ["are", "you", "robot"],
            ["who", "are", "you"],
        ],
    }

    # 将手动添加的意图插入到数据中
    def insert_manually(target, prototype):
        """接收标记化文档原型以重复，直到获得长度为 target"""
        factor = math.ceil(target / len(prototype))
        print(factor)
        content = prototype * factor
        return [content[i] for i in range(target)]

    # 更新训练数据
    for intent_name in manually_added_intents.keys():
        train[intent_name] = insert_manually(
            1000, [*manually_added_intents[intent_name]]
        )

    # 3. 添加混合意图

    hybrid_intents = {
        "update": (
            300,
            700,
            [
                ["want", "update"],
                ["update", "not", "working"],
                ["phone", "need", "update"],
            ],
            intents_tags["update"],
        ),
        "info": (
            800,
            200,
            [
                ["need", "information"],
                ["want", "to", "know", "about"],
                ["what", "are", "macbook", "stats"],
                ["any", "info", "next", "release", "?"],
            ],
            intents_tags["info"],
        ),
        "payment": (
            300,
            700,
            [
                ["payment", "not", "through"],
                ["iphone", "apple", "pay", "but", "not", "arrive"],
                ["how", "pay", "for", "this"],
                ["can", "i", "pay", "for", "this", "first"],
            ],
            intents_tags["payment"],
        ),
        "forgot_password": (
            600,
            400,
            [
                ["forgot", "my", "pass"],
                ["forgot", "my", "login", "details"],
                ["cannot", "log", "in", "password"],
                ["lost", "account", "recover", "password"],
            ],
            intents_tags["forgot_password"],
        ),
    }

    def insert_hybrid(manual_target, generated_target, prototype, itag):
        return insert_manually(manual_target, prototype) + list(
            generate_intent(generated_target, itag)
        )

    # 更新训练数据
    for intent_name, args in hybrid_intents.items():
        train[intent_name] = insert_hybrid(*args)

    # 4. 将宽格式转换为长格式，以便我的 NN 模型可以读取 - 并进行整理
    neat_train = (
        pd.DataFrame(train.T.unstack())
        .reset_index()
        .iloc[:, 1:]
        .rename(columns={"level_1": "Intent", 0: "Utterance"})
    )
    # 重新排序
    neat_train = neat_train[["Utterance", "Intent"]]

    # 5. 将此原始训练数据保存为序列化文件
    neat_train.to_pickle("../objects/train.pkl")

    # 显示样式
    show = (
        lambda x: x.style.set_properties(
            **{
                "background-color": "black",
                "color": "lawngreen",
                "border-color": "white",
            }
        )
        .applymap(lambda x: f"color: {'lawngreen' if isinstance(x,str) else 'red'}")
        .background_gradient(cmap="Blues")
    )

    st.header("训练数据 - 比较不同意图视图")
    st.dataframe(show(train))

    st.header("格式化为模型输入的训练数据")
    st.dataframe(show(neat_train))

    """
    意图评估
    """

    st.subheader("查看每个意图的前几个词")
    # 将单词排名表数据框存储在此字典中
    wordranks = {}

    # 对每个意图进行评估 - 根据文本数据中的计数绘制前 10 个词
    def top10_bagofwords(data, output_name, title):
        """接收数据并绘制基于计数的前 10 个词"""
        bagofwords = CountVectorizer()
        # 输出将是一个稀疏矩阵
        inbound = bagofwords.fit_transform(data)
        # 检查缩写和口语的使用频率
        word_counts = np.array(np.sum(inbound, axis=0)).reshape((-1,))
        words = np.array(bagofwords.get_feature_names())
        words_df = pd.DataFrame({"word": words, "count": word_counts})
        words_rank = words_df.sort_values(by="count", ascending=False)
        wordranks[output_name] = words_rank
        # words_rank.to_csv('words_rank.csv') # 将其存储在 csv 中，以便自行检查和查看
        # 可视化前 10 个词
        plt.figure(figsize=(12, 6))
        sns.barplot(
            words_rank["word"][:10],
            words_rank["count"][:10].astype(str),
            palette="inferno",
        )
        plt.title(title)

        # Saving
        # plt.savefig(f'visualizations/next_ver/{output_name}.png')
        st.pyplot()

    # Doing my bucket evaluations here - seeing what each distinct bucket intent means
    for i in train.columns:
        top10_bagofwords(
            train[i].apply(" ".join), f"bucket_eval/{i}", f"Top 10 Words in {i} Intent",
        )


if __name__ == "__main__":
    main()

