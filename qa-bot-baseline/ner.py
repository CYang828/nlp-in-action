# 数据科学
import pandas as pd
import numpy as np
import sklearn

# 命名实体识别
import spacy
from spacy import displacy
import random
from spacy.matcher import PhraseMatcher
from pathlib import Path

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)
import collections
import yaml
import pickle
import streamlit as st
import imgkit

# 读取意图
with open(r"../objects/intents.yml") as file:
    intents = yaml.load(file, Loader=yaml.FullLoader)

# 读取训练数据
train = pd.read_pickle("../objects/train.pkl")

# 读取处理后的数据
processed = pd.read_pickle("../objects/processed.pkl")

# 读取我们训练好的模型
hardware_nlp = pickle.load(open("../models/hardware_big_nlp.pkl", "rb"))
app_nlp = pickle.load(open("../models/app_big_nlp.pkl", "rb"))

# 用于显示我的displacy可视化的包装器
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

# 测试结果
test_text_hardware = "My iphone sucks but my macbook pro doesnt. Why couldnt they make\
            my iphone better. At least I could use airpods with it. Mcabook pro is\
            the best! Apple watches too. Maybe if they made the iphone more like the\
            ipad or my TV it would be alright. Mac. Ugh."
test_text_app = "My top favorite apps include the facetime application, the apple books on my iphone, and the podcasts\
        application. Sometimes instead of spotify I would listen to apple music. My macbook is running\
        Catalina btw."


def extract_hardware(user_input, visualize=False):
    """接受用户输入，并输出提取的所有实体。还制作了一个用于使用displacy进行可视化的切换器。"""
    # 加载
    hardware_nlp = pickle.load(open("../models/hardware_big_nlp.pkl", "rb"))
    doc = hardware_nlp(user_input)

    extracted_entities = []

    # 这些是您可以取出的对象
    for ent in doc.ents:
        extracted_entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))

    # 如果要进行可视化
    if visualize == True:
        # 使用displacy进行实体标记的可视化（运行服务器）
        colors = {"HARDWARE": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
        options = {"ents": ["HARDWARE"], "colors": colors}
        # 保存为HTML字符串
        html = displacy.render(doc, style="ent", options=options)
        # with open("displacy/hardware.html", "a") as out:
        #     out.write(html + "\n")
        # 双换行似乎会影响渲染
        html = html.replace("\n\n", "\n")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    return extracted_entities


def extract_app(user_input, visualize=False):
    """接受用户输入，并输出提取的所有实体。还制作了一个用于使用displacy进行可视化的切换器。"""
    # 加载
    app_nlp = pickle.load(open("../models/app_big_nlp.pkl", "rb"))
    doc = app_nlp(user_input)

    extracted_entities = []

    # 这些是您可以取出的对象
    for ent in doc.ents:
        extracted_entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))

    # 如果要进行可视化
    if visualize == True:
        # 使用displacy进行实体标记的可视化（运行服务器）
        colors = {"APP": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
        options = {"ents": ["APP"], "colors": colors}
        html = displacy.render(doc, style="ent", options=options)
        # with open("displacy/hardware.html", "a") as out:
        #     out.write(html + "\n")
        # 双换行似乎会影响渲染
        html = html.replace("\n\n", "\n")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    return extracted_entities


def extract_default(user_input):
    pass


# 测试功能
# print(extract_app(test_text_app))
# extract_hardware(test_text_hardware, visualize=True)
