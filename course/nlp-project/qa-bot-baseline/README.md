# 苹果公司客服机器人项目

## 运行

```bash
python generate_ner_models.py
```

```bash
streamlit run app.py
```

## 项目结构

```
deploy/
│
├── README.md
├── actions.py                      （识别到意图对应动作）
├── app.py                          （项目入口）
├── bot.py                          （对话机器人代码）
├── generate_ner_models.py           (生成ner模型)
├── generate_train.py               （生成训练数据）
├── initialize_intent_classifier.py （初始化意图分类函数）
├── intent_classifier.py            （意图分类模型）
├── keyword_exploration.py          （关键字探索）
└── ner.py                          （命名实体识别）
```

* **actions.py**

这个类列举了用户可以执行的所有操作。根据我的Keras模型预测的意图和spaCy NER提取的实体，它将把它映射到一个动作，并向用户返回一个响应。

* **app.py**

将所有其他脚本编译在一起。

* **bot.py**

包含聊天机器人页面（即创建Eve机器人的大部分逻辑）。其他页面可以在app.py中访问。

* **generate_train.py**

这是一个Streamlit工具，我发现Streamlit的可视化对于使用Doc2Vec生成训练数据并将其输出到文件非常有帮助。

* **initialize_intent_classifier.py**

这是意图分类器的简化版本，只涉及加载模型，以便我的应用程序可以根据用户输入获取模型预测。我们假设模型已经保存到文件中，因此我们可以直接加载它。

* **intent_classifier.py**

这是完整的意图分类器，它接受数据并实际运行模型步骤，以实现最终的模型输出。

* **keyword_exploration.py**

创建了一个Streamlit工具，用于使用关键字或它们的组合在数据集中探索意图。

* **ner.py**

命名实体识别工具。


## 精进方向

- 多轮对话 -- 上下文建模
- 知识的延展（行业知识）-- 知识图谱
- 语义理解的精进  -- 大模型

## 参考

[streamlit 文档](https://docs.streamlit.io/)


