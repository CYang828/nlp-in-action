import streamlit as st
import pandas as pd
import numpy as np

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)
# 使可视化更美观
sns.set_style("whitegrid")
# 组合探索
import itertools
import yaml

# 加载处理后的数据
processed = pd.read_pickle("../objects/processed.pkl")
processed["Real Inbound"] = [[i] for i in processed["Real Inbound"]]
processed["Real Outbound"] = [[i] for i in processed["Real Outbound"]]


def main():

    st.title("可视化我的处理过的 Twitter 数据中意图的分布")

    # select = st.multiselect("选择你的数据集")

    # 输入意图

    """
    关键词搜索
    """

    # 通过关键词搜索（单个关键词过滤器）
    keyword = st.text_input("今天你想探索哪个关键词?")

    # 查看处理过的推文是什么样子
    filt = [
        (i, j) for i, j in enumerate(processed["Processed Inbound"]) if keyword in j
    ]
    filtered = processed.iloc[[i[0] for i in filt]]

    # 显示包含关键词的推文数量
    st.text(f"有{len(filtered)}条推文包含关键词{keyword}")

    st.subheader(f"这里是包含关键词的推文")
    # 显示包含关键词的数据框
    pd.set_option("display.max_columns", None)
    st.dataframe(filtered.iloc[:, 0])
    pd.set_option("display.max_columns", None)
    st.dataframe(filtered.iloc[:, 1])
    pd.set_option("display.max_columns", None)
    st.dataframe(filtered.iloc[:, 2])

    """
    意图探索
    """

    st.subheader("数据中意图的分布")

    intents = {
        "update": ["update"],
        "battery": ["battery", "power"],
        "forgot_password": ["password", "account", "login"],
        "repair": ["repair", "fix", "broken"],
        "payment": ["credit", "card", "payment", "pay"],
    }

    st.write(intents)

    def get_key_tweets(series, keywords):
        """ 输入关键词列表，输出包含至少一个这些关键词的推文 """
        keyword_tweets = []
        for tweet in series:
            # 检查关键词是否在推文中
            for keyword in keywords:
                if keyword in tweet:
                    keyword_tweets.append(tweet)
        return keyword_tweets

    def to_set(l):
        """ 为了将推文转换为集合以检查交集，我们需要将它们变为不可变的元组，因为集合只接受不可变元素 """
        return set([tuple(row) for row in l])

    # 使用上面的函数来可视化数据集中意图的分布
    intent_lengths = [
        len(get_key_tweets(processed["Processed Inbound"], intents[intent]))
        for intent in intents.keys()
    ]
    keyword = pd.DataFrame(
        {"intents": list(intents.keys()), "intent_lengths": intent_lengths}
    ).sort_values("intent_lengths", ascending=False)

    # 可视化
    plt.figure(figsize=(9, 7))
    plt.bar(keyword["intents"], keyword["intent_lengths"], color="#00acee")
    plt.title("使用关键词搜索的意图分布")
    plt.xlabel("意图")
    plt.xticks(rotation=90)
    plt.ylabel("具有意图关键词的推文数量")
    st.pyplot(bbox_inches="tight")

    # 比例可视化
    plt.figure(figsize=(9, 7))
    plt.bar(
        keyword["intents"], keyword["intent_lengths"] * 100 / 75879, color="#00acee"
    )
    plt.title("使用关键词搜索的意图分布")
    plt.xlabel("意图")
    plt.xticks(rotation=90)
    plt.ylabel("具有意图关键词的推文百分比")
    st.pyplot(bbox_inches="tight")

    """
    探索意图的组合
    """

    # 初始化所有组合出现最小数量的阈值
    thres = [500, 10, 5, 5]

    # 意图推文具有所有键，值包含包含该键的所有推文，作为集合
    intent_tweets = {}
    for key in intents.keys():
        intent_tweets[key] = to_set(
            get_key_tweets(processed["Processed Inbound"], intents[key])
        )

    # 遍历所有配对，并获取配对之间的推文交集数量
    keyword_overlaps = {}

    # 2的组合

    # 每个i返回一个长度为r的元组，这里r为2
    for i in list(itertools.combinations(list(intents.keys()), 2)):
        a = to_set(intent_tweets[i[0]])
        b = to_set(intent_tweets[i[1]])
        # 将配对插入字典
        keyword_overlaps[f"{i[0]} + {i[1]}"] = len(a.intersection(b))

    # 过滤为仅包含重要的配对，我定义为大于100
    combs = []
    counts = []
    for i in keyword_overlaps.items():
        if i[1] > thres[0]:
            combs.append(i[0])
            counts.append(i[1])

    # 也可视化
    v = pd.DataFrame({"Combination": combs, "Counts": counts}).sort_values(
        "Counts", ascending=False
    )
    plt.figure(figsize=(9, 6))
    sns.barplot(x=v["Combination"], y=v["Counts"], palette="magma")
    plt.title(f"2个关键词的组合（至少{thres[0]}次出现）")
    plt.xticks(rotation=90)
    st.pyplot(bbox_inches="tight")

    # 3的组合
    keyword_overlaps = {}

    try:
        # 每个i返回一个长度为r的元组，这里r为3
        for i in list(itertools.combinations(list(intents.keys()), 3)):
            a = to_set(intent_tweets[i[0]])
            b = to_set(intent_tweets[i[1]])
            c = to_set(intent_tweets[i[2]])
            # 将配对插入字典
            keyword_overlaps[f"{i[0]} + {i[1]} + {i[2]}"] = len(
                a.intersection(b).intersection(c)
            )

        # 过滤为仅包含重要的配对，我定义为大于10
        combs = []
        counts = []
        for i in keyword_overlaps.items():
            if i[1] > thres[1]:
                combs.append(i[0])
                counts.append(i[1])

        # 也可视化
        v = pd.DataFrame({"Combination": combs, "Counts": counts}).sort_values(
            "Counts", ascending=False
        )
        plt.figure(figsize=(9, 6))
        sns.barplot(x=v["Combination"], y=v["Counts"], palette="magma")
        plt.title(f"3个关键词的组合（至少{thres[1]}次出现）")
        plt.xticks(rotation=90)
        st.pyplot(bbox_inches="tight")
    except ValueError as e:
        st.text(f"3个组合不够（阈值 = {thres[1]})")

    # 4的组合
    keyword_overlaps = {}

    try:
        # 每个i返回一个长度为r的元组，这里r为4
        for i in list(itertools.combinations(list(intents.keys()), 4)):
            a = to_set(intent_tweets[i[0]])
            b = to_set(intent_tweets[i[1]])
            c = to_set(intent_tweets[i[2]])
            d = to_set(intent_tweets[i[3]])
            # 将配对插入字典
            keyword_overlaps[f"{i[0]} + {i[1]} + {i[2]} + {i[3]}"] = len(
                a.intersection(b).intersection(c).intersection(d)
            )

        # 过滤为仅包含重要的配对，我定义为大于5
        combs = []
        counts = []
        for i in keyword_overlaps.items():
            if i[1] > thres[2]:
                combs.append(i[0])
                counts.append(i[1])

        # 也可视化
        v = pd.DataFrame({"Combination": combs, "Counts": counts}).sort_values(
            "Counts", ascending=False
        )
        plt.figure(figsize=(9, 6))
        sns.barplot(x=v["Combination"], y=v["Counts"], palette="magma")
        plt.title(f"4个关键词的组合（至少{thres[2]}次出现）")
        plt.xticks(rotation=90)
        st.pyplot(bbox_inches="tight")
    except ValueError as e:
        st.text(f"4个组合不够（阈值 = {thres[2]})")

    # 5的组合
    keyword_overlaps = {}

    try:
        # 每个i返回一个长度为r的元组，这里r为5
        for i in list(itertools.combinations(list(intents.keys()), 5)):
            a = to_set(intent_tweets[i[0]])
            b = to_set(intent_tweets[i[1]])
            c = to_set(intent_tweets[i[2]])
            d = to_set(intent_tweets[i[3]])
            e = to_set(intent_tweets[i[4]])
            # 将配对插入字典
            keyword_overlaps[f"{i[0]} + {i[1]} + {i[2]} + {i[3]} + {i[4]}"] = len(
                a.intersection(b).intersection(c).intersection(d).intersection(e)
            )

        # 过滤为仅包含重要的配对，我定义为大于5
        combs = []
        counts = []
        for i in keyword_overlaps.items():
            if i[1] > thres[3]:
                combs.append(i[0])
                counts.append(i[1])

        # 也可视化
        v = pd.DataFrame({"Combination": combs, "Counts": counts}).sort_values(
            "Counts", ascending=False
        )
        plt.figure(figsize=(9, 6))
        sns.barplot(x=v["Combination"], y=v["Counts"], palette="magma")
        plt.title(f"5个关键词的组合（至少{thres[3]}次出现）")
        plt.xticks(rotation=90)
        st.pyplot(bbox_inches="tight")
    except ValueError as e:
        st.text(f"5个组合不够（阈值 = {thres[3]}）")


if __name__ == "__main__":
    main()
