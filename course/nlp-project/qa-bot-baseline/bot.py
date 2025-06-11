import streamlit as st
from actions import Actions
from ner import extract_app, extract_hardware
from initialize_intent_classifier import infer_intent
import pandas as pd
import numpy as np
import yaml


# 可视化
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

# 加载实体
with open(r"../objects/entities.yml") as file:
    entities = yaml.load(file, Loader=yaml.FullLoader)

# 加载预测
# with open(r"../objects/sorted_predictions.yml") as file:
#     sorted_predictions = yaml.load(file, Loader=yaml.FullLoader)

# 加载训练数据
train = pd.read_pickle("../objects/train.pkl")

sns.set(style="ticks", color_codes=True)
sns.set_style(style="whitegrid")

# 响应模板
respond = lambda response: f'苹果客服: :green[**{response}**]'


def main(phrase="有什么问题你尽管问!"):
    # 为这次对话实例化Action对象
    a = Actions(phrase)

    # st.text(respond(a.utter_greet()))

    intents, user_input, history_df, end = conversation(a)
    print(end)

    if st.sidebar.button("显示推理过程"):
        backend_dash(intents, user_input, history_df)

    if end == False:
        st.cache_data.clear()
        # 意图澄清
        conversation(Actions("你能换种说法吗?"))


def conversation(starter):
    """代表一整个对话流程，接受Actions对象以了解从哪个提示开始"""

    a = starter

    user_input, hardware, app, intents, history_df = talk(prompt=a.startup)

    # 存储当前状态
    max_intent, extracted_entities = action_mapper(history_df)

    if extracted_entities != []:
        if len(extracted_entities) == 1:
            entity = extracted_entities[0]
            print(f"找到1个实体: {entity}")
        elif len(extracted_entities) == 2:
            entity = extracted_entities[:2]
            print(f"找到2个实体: {entity}")
    else:
        entity = None

    end = listener(max_intent, entity, a)

    return (intents, user_input, history_df, end)


def talk(prompt):
    """经历并开始一次对话并返回:

    User_input: 字符串
    Hardware: 包含提取的实体的字符串列表
    App: 包含提取的实体的字符串列表
    Intents: 可以解包为的元组:
        - User_input
        - Predictions: 包含意图为键和预测概率（0-1）为值的字典
    History_df: 给定输入的对话状态
    """
    # 输入框
    user_input = st.text_input(prompt)

    # 预测意图并识别实体
    intents, hardware, app = initialize(user_input)
    user_input, prediction = intents

    # 初始化历史记录
    columns = entities["hardware"] + entities["apps"] + list(prediction.keys())
    history_df = pd.DataFrame(dict(zip(columns, np.zeros(len(columns)))), index=[0])

    # 转换为对话历史条目，然后将其附加到数据框中
    history_df = history_df._append(to_row(prediction, hardware, app), ignore_index=True)

    return (user_input, hardware, app, intents, history_df)


def listener(max_intent, entity, actions):
    """接受对话状态并将其映射到响应"""

    # 用于后续的嵌套函数
    def follow_up(prompt="你能换种说法吗?"):
        """后续的业务逻辑"""

        # 了解对话是否已结束的布尔值
        end = None

        st.markdown("**这解决了你的问题吗?**")
        col1, col2 = st.columns(2)
        with col1:
            yes = st.button("是", type="primary")
        with col2:
            no = st.button("否", type="primary")

        if yes:
            st.markdown(respond("太好了!很高兴能为你提供服务!"))
            end = True

        if no:
            # 继续下一次对话
            end = False
        return end

    # 初始化actions对象
    a = actions

    # 初始化end
    end = None

    if max_intent == "greeting":
        st.markdown(respond(a.utter_greet()))
    elif max_intent == "info":
        st.markdown(respond(a.info(entity)))
        end = follow_up()
    elif max_intent == "update":
        st.markdown(respond(a.update(entity)))
        end = follow_up()
    elif max_intent == "forgot_password":
        st.markdown(respond(a.forgot_pass()))
        end = follow_up()
    elif max_intent == "challenge_robot":
        st.markdown(respond(a.challenge_robot()))
    elif max_intent == "goodbye":
        st.markdown(respond(a.utter_goodbye()))
        # st.image("images/eve-bye.jpg", width=400)
        st.markdown("果bo向您挥手告别！")
    elif max_intent == "payment":
        st.markdown(respond(a.payment()))
        end = follow_up()
    elif max_intent == "speak_representative":
        st.markdown(respond(a.link_to_human()))
        st.image("images/representative.png")
    elif max_intent == "battery":
        st.markdown(respond(a.battery(entity)))
        end = follow_up()
    elif max_intent == "fallback":
        st.markdown(respond(a.fallback()))

    return end


def backend_dash(intents, user_input, history_df):
    """使用仪表板可视化给定状态参数的整个对话状态"""
    # 显示预测
    st.subheader("机器人预测")
    # 进一步解包
    user_input, pred = intents
    pred = {k: round(float(v), 3) for k, v in pred.items()}

    # 可视化意图分类
    g = sns.barplot(
        list(pred.keys()),
        list(pred.values()),
        palette=sns.cubehelix_palette(8, reverse=True),
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    st.pyplot(bbox_inches="tight")

    # 捕获的实体
    st.subheader("检测到的硬件实体")
    hardware = extract_hardware(user_input, visualize=True)
    st.subheader("检测到的应用实体")
    app = extract_app(user_input, visualize=True)

    # 显示历史
    st.subheader("对话状态历史")
    st.dataframe(history_df)


def to_row(prediction, hardware, app):
    row = []

    # 硬件
    if hardware == []:
        for i in range(len(entities["hardware"])):
            row.append(0)
    else:
        for entity in entities["hardware"]:
            if hardware[0][0] == entity:
                row.append(1)
            else:
                row.append(0)

    # 应用
    if app == []:
        for i in range(len(entities["apps"])):
            row.append(0)
    else:
        for entity in entities["apps"]:
            if app[0][0] == entity:
                row.append(1)
            else:
                row.append(0)

    # 预测 - 插入所有概率
    for i in prediction.items():
        row.append(i[1])

    # 转换为数据框
    columns = entities["hardware"] + entities["apps"] + list(prediction.keys())
    df = pd.DataFrame(dict(zip(columns, row)), index=[0])

    return df


def action_mapper(history_df):
    """简单地将历史状态映射到:

    最大意图: 字符串
    实体: 提取的实体列表

    """
    prediction_probs = history_df.iloc[-1:, -len(set(train["Intent"])) :]
    predictions = [
        *zip(list(prediction_probs.columns), list(prediction_probs.values[0]))
    ]

    # 找到实体
    entities = history_df.iloc[-1:, : -len(set(train["Intent"]))]
    mask = [True if i == 1.0 else False for i in list(entities.values[0])]
    extracted_entities = [b for a, b in zip(mask, list(entities.columns)) if a]

    # 通过排序找到最大意图
    predictions.sort(key=lambda x: x[1])
    # 取最大值
    max_tuple = predictions[-1:]
    # 最大意图
    #     print(max_tuple)
    max_intent = max_tuple[0][0]
    #     print(f'max_intent{max_intent}')

    # TODO: 如果置信度分数不够高，则回退
    # 意图澄清

    return (max_intent, extracted_entities)


def initialize(user_input):
    """接受用户输入并返回实体表示和预测的意图"""
    # 意图分类
    intents = infer_intent(user_input)

    # NER
    hardware = extract_hardware(user_input)
    app = extract_app(user_input)

    if hardware == []:
        hardware = "none"

    if app == []:
        app = "none"

    return (intents, hardware, app)


if __name__ == "__main__":
    main()
