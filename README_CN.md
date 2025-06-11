# NLP 实战

## 关于这个项目

**NLP 实战** 是一个全面的、动手实践的学习资源，旨在帮助你通过实际应用掌握自然语言处理技术。本项目通过提供分步教程、代码示例和完整项目，架起了NLP理论与实际应用之间的桥梁。

### 🎯 你将学到什么

- **基础知识**：从基本文本处理到高级神经语言模型
- **实用技能**：真实世界的NLP流水线开发和部署
- **现代技术**：Transformer模型、预训练模型和大语言模型
- **行业应用**：意图分类、命名实体识别和对话AI
- **最佳实践**：数据预处理、模型评估和生产部署

### 🌟 核心特色

- **渐进式学习路径**：从入门到高级的结构化课程
- **实践导向**：通过构建真实项目和提供的数据集来学习
- **现代技术栈**：TensorFlow 2、Hugging Face Transformers 和 LangChain
- **完整项目**：可直接用于生产的端到端实现
- **丰富资源**：理论解释、代码笔记本和可下载资源

### 👥 适合人群

- **数据科学家** 希望专精于NLP领域
- **软件工程师** 有兴趣为应用程序添加NLP功能
- **学生和研究人员** 寻求实用的NLP经验
- **机器学习工程师** 想要了解现代NLP部署策略

### 🚀 学习方法

每个模块结合了：
- **理论**：概念和算法的详细解释
- **实践**：带有可运行代码示例的Jupyter笔记本
- **项目**：真实世界的应用和案例研究
- **资源**：数据集、预训练模型和参考论文

通过任务实战，学习NLP。

gitee 备份地址 https://gitee.com/cyang828/nlp-in-action

## ⚡ 快速开始

### 克隆项目

使用以下命令递归克隆项目，包含所有子模块：

```bash
git clone --recursive https://github.com/CYang828/nlp-in-action.git
cd nlp-in-action
```

如果你已经克隆了仓库但没有使用 `--recursive` 参数，可以使用以下命令初始化和更新子模块：

```bash
git submodule update --init --recursive
```

### 环境配置

1. **创建虚拟环境**（推荐）：
```bash
python -m venv nlp-env
source nlp-env/bin/activate  # Windows 系统: nlp-env\Scripts\activate
```

2. **安装依赖**：
```bash
pip install -r requirements.txt
```

3. **启动 Jupyter Notebook**：
```bash
jupyter notebook
```

4. **从第一个教程开始**：打开 `0.first-pipeline.ipynb` 开始你的 NLP 学习之旅！

### 备用安装方式

你也可以从 Gitee 镜像克隆：
```bash
git clone --recursive https://gitee.com/cyang828/nlp-in-action.git
```

## 提问
- FAQ 文档地址：https://rarelimiting.feishu.cn/wiki/EQTRwntmnidkpSkK9pOcLzOcnze


## 学习路线图
| 课程名称 | 理论讲解 | 下载 | 笔记 | 论文 |
| -------- |  --------- | --------- | -------- | -------- |
| 第一个自然语言Pipeline | [阅读](https://rarelimiting.feishu.cn/wiki/CTkPw9uOdiAZrrkSaXucJt1SnBh?fromScene=spaceOverview&create_from=create_doc_to_wiki) | | [地址](https://github.com/CYang828/nlp-in-action/blob/main/0.first-pipeline.ipynb) | |
| EDA 和预处理 | [阅读](https://rarelimiting.feishu.cn/wiki/XWyfwGJR2iupNbkBS1ycIHU7n6o?fromScene=spaceOverview) | [数据下载地址](https://aimaksen.rarelimiting.com/twitter-twcs.zip) [模型下载地址](https://aimaksen.rarelimiting.com/models.zip) | [地址](https://github.com/CYang828/nlp-in-action/blob/main/01.eda%26preprocessing.ipynb) | |
| 文本表示 | [阅读](https://rarelimiting.feishu.cn/wiki/P35awju87i1fobkxYMkcTqF0nNf?fromScene=spaceOverview) | [中间数据下载地址](https://aimaksen.rarelimiting.com/objects.zip) | [地址](https://github.com/CYang828/nlp-in-action/blob/main/02.text-representation.ipynb) | |
| word2vec 复现 | [阅读](https://rarelimiting.feishu.cn/wiki/QtNrwjJimiio5jk4twCcmDtjn5e?fromScene=spaceOverview) [TensorFlow 2 学习指南](https://rarelimiting.feishu.cn/wiki/GriywObmEixwF4ksELncLXgrnFe?fromScene=spaceOverview) | | [地址](https://github.com/CYang828/nlp-in-action/blob/main/03.word2vec.ipynb) | [地址](https://arxiv.org/pdf/1301.3781.pdf) |
| 意图探索 | | | [地址](https://github.com/CYang828/nlp-in-action/blob/main/04.explore-intents.ipynb) | |
| 意图分类 | | | [地址](https://github.com/CYang828/nlp-in-action/blob/main/05.intent-classfication.ipynb) | |
| 实体识别 | | | [地址](https://github.com/CYang828/nlp-in-action/blob/main/06.ner.ipynb) | |
| 完整项目 |  | | [项目入口](deploy/README.md) | |
| 预训练模型和大模型的进化历程 |  | | [Transformer笔记](https://github.com/CYang828/transformer-all-in-one/tree/dd206218ba169880e27803ddc5ad59ac3cbd25b5) [阅读](https://rarelimiting.feishu.cn/wiki/ZaXowgBbTiAm7HkotLNczmq3nsb?fromScene=spaceOverview)  | |
| Huggingface Transformer 指南 |  | | [Huggingface笔记](https://github.com/CYang828/nlp-in-action/tree/main/huggingface-tutorials) |  |
| 大模型应用开发 |  | | [大模型开发笔记](https://rarelimiting.feishu.cn/wiki/PW1fw9iQviRbznkizgzcgbtHnRg?fromScene=spaceOverview) [Langchain开发指南](https://github.com/CYang828/llm-tutorials-chinese/tree/58c0a55aea3efb9f488ada8267b79ad43a400f71/book)  |  |
| 大模型应用开发高级技巧 |  | | [大模型开发笔记](https://rarelimiting.feishu.cn/wiki/PW1fw9iQviRbznkizgzcgbtHnRg?fromScene=spaceOverview) [Langchain开发指南](https://github.com/CYang828/llm-tutorials-chinese/tree/58c0a55aea3efb9f488ada8267b79ad43a400f71/book)  |  |
| 提示工程 |  | | [提示工程笔记](https://rarelimiting.feishu.cn/wiki/PruLw3hjMiHZFxkepO8cfFSBnWh?from=from_copylink) |   |
| 知识图谱 |  | | [知识图谱笔记](https://rarelimiting.feishu.cn/wiki/ZT1hwsfrniwySQkncl4c1NQEn3d?from=from_copylink) |   |


