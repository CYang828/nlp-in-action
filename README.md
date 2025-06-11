# NLP in Action

[中文版本](README_CN.md) | English Version

## About This Project

**NLP in Action** is a comprehensive, hands-on learning resource designed to help you master Natural Language Processing through practical implementation. This project bridges the gap between NLP theory and real-world applications by providing step-by-step tutorials, code examples, and complete projects.

### 🎯 What You'll Learn

- **Fundamentals**: From basic text processing to advanced neural language models
- **Practical Skills**: Real-world NLP pipeline development and deployment
- **Modern Techniques**: Transformer models, pre-trained models, and large language models
- **Industry Applications**: Intent classification, named entity recognition, and conversational AI
- **Best Practices**: Data preprocessing, model evaluation, and production deployment

### 🌟 Key Features

- **Progressive Learning Path**: Structured curriculum from beginner to advanced levels
- **Hands-on Approach**: Learn by building real projects with provided datasets
- **Modern Stack**: TensorFlow 2, Hugging Face Transformers, and LangChain
- **Complete Projects**: End-to-end implementations ready for production
- **Rich Resources**: Theory explanations, code notebooks, and downloadable assets

### 👥 Who This Is For

- **Data Scientists** looking to specialize in NLP
- **Software Engineers** interested in adding NLP capabilities to applications
- **Students and Researchers** seeking practical NLP experience
- **ML Engineers** wanting to understand modern NLP deployment strategies

### 🚀 Learning Approach

Each module combines:
- **Theory**: Detailed explanations of concepts and algorithms
- **Practice**: Jupyter notebooks with runnable code examples
- **Projects**: Real-world applications and case studies
- **Resources**: Datasets, pre-trained models, and reference papers

Gitee backup address: https://gitee.com/cyang828/nlp-in-action

## ⚡ Quick Start

### Clone the Repository

Use the following command to recursively clone the project with all submodules:

```bash
git clone --recursive https://github.com/CYang828/nlp-in-action.git
cd nlp-in-action
```

If you've already cloned the repository without the `--recursive` flag, you can initialize and update submodules with:

```bash
git submodule update --init --recursive
```

### Environment Setup

1. **Create a virtual environment** (recommended):
```bash
python -m venv nlp-env
source nlp-env/bin/activate  # On Windows: nlp-env\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**:
```bash
jupyter notebook
```

4. **Start with the first tutorial**: Open `0.first-pipeline.ipynb` to begin your NLP journey!

### Alternative Installation

You can also clone from the Gitee mirror:
```bash
git clone --recursive https://gitee.com/cyang828/nlp-in-action.git
```

## Q&A
- FAQ documentation: https://rarelimiting.feishu.cn/wiki/EQTRwntmnidkpSkK9pOcLzOcnze

## Learning Roadmap
| Course Name | Theory Explanation | Download | Notes | Papers |
| -------- |  --------- | --------- | -------- | -------- |
| First Natural Language Pipeline | [Read](https://rarelimiting.feishu.cn/wiki/CTkPw9uOdiAZrrkSaXucJt1SnBh?fromScene=spaceOverview&create_from=create_doc_to_wiki) | | [Link](https://github.com/CYang828/nlp-in-action/blob/main/0.first-pipeline.ipynb) | |
| EDA and Preprocessing | [Read](https://rarelimiting.feishu.cn/wiki/XWyfwGJR2iupNbkBS1ycIHU7n6o?fromScene=spaceOverview) | [Data Download](https://aimaksen.rarelimiting.com/twitter-twcs.zip) [Model Download](https://aimaksen.rarelimiting.com/models.zip) | [Link](https://github.com/CYang828/nlp-in-action/blob/main/01.eda%26preprocessing.ipynb) | |
| Text Representation | [Read](https://rarelimiting.feishu.cn/wiki/P35awju87i1fobkxYMkcTqF0nNf?fromScene=spaceOverview) | [Intermediate Data Download](https://aimaksen.rarelimiting.com/objects.zip) | [Link](https://github.com/CYang828/nlp-in-action/blob/main/02.text-representation.ipynb) | |
| Word2vec Implementation | [Read](https://rarelimiting.feishu.cn/wiki/QtNrwjJimiio5jk4twCcmDtjn5e?fromScene=spaceOverview) [TensorFlow 2 Learning Guide](https://rarelimiting.feishu.cn/wiki/GriywObmEixwF4ksELncLXgrnFe?fromScene=spaceOverview) | | [Link](https://github.com/CYang828/nlp-in-action/blob/main/03.word2vec.ipynb) | [Link](https://arxiv.org/pdf/1301.3781.pdf) |
| Intent Exploration | | | [Link](https://github.com/CYang828/nlp-in-action/blob/main/04.explore-intents.ipynb) | |
| Intent Classification | | | [Link](https://github.com/CYang828/nlp-in-action/blob/main/05.intent-classfication.ipynb) | |
| Named Entity Recognition | | | [Link](https://github.com/CYang828/nlp-in-action/blob/main/06.ner.ipynb) | |
| Complete Project |  | | [Project Entry](deploy/README.md) | |
| Pre-trained Models and Evolution of Large Language Models |  | | [Transformer Notes](https://github.com/CYang828/transformer-all-in-one/tree/dd206218ba169880e27803ddc5ad59ac3cbd25b5) [Read](https://rarelimiting.feishu.cn/wiki/ZaXowgBbTiAm7HkotLNczmq3nsb?fromScene=spaceOverview)  | |
| Hugging Face Transformers Guide |  | | [Hugging Face Notes](https://github.com/CYang828/nlp-in-action/tree/main/huggingface-tutorials) |  |
| Large Language Model Application Development |  | | [LLM Development Notes](https://rarelimiting.feishu.cn/wiki/PW1fw9iQviRbznkizgzcgbtHnRg?fromScene=spaceOverview) [Langchain Development Guide](https://github.com/CYang828/llm-tutorials-chinese/tree/58c0a55aea3efb9f488ada8267b79ad43a400f71/book)  |  |
| Advanced Techniques for LLM Application Development |  | | [LLM Development Notes](https://rarelimiting.feishu.cn/wiki/PW1fw9iQviRbznkizgzcgbtHnRg?fromScene=spaceOverview) [Langchain Development Guide](https://github.com/CYang828/llm-tutorials-chinese/tree/58c0a55aea3efb9f488ada8267b79ad43a400f71/book)  |  |
| Prompt Engineering |  | | [Prompt Engineering Notes](https://rarelimiting.feishu.cn/wiki/PruLw3hjMiHZFxkepO8cfFSBnWh?from=from_copylink) |   |
| Knowledge Graph |  | | [Knowledge Graph Notes](https://rarelimiting.feishu.cn/wiki/ZT1hwsfrniwySQkncl4c1NQEn3d?from=from_copylink) |   | 