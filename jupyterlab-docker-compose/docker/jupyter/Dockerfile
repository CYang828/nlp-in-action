FROM quay.io/jupyter/minimal-notebook:latest

# 切换到root用户安装系统包
USER root

# 安装编译工具和开发依赖
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        gcc \
        g++ \
        python3-dev \
        git \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 切换回默认用户
USER $NB_UID

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    pip install -r /tmp/requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装代码格式化工具
RUN pip install black autopep8 isort yapf -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装JupyterLab扩展和插件
RUN pip install jupyterlab-code-formatter \
                jupyterlab-drawio \
                jupyterlab-system-monitor \
                jupyterlab-topbar \
                jupyterlab-theme-solarized-dark \
                jupyterlab-night \
                jupyterlab-markup \
                jupyter-ai \
    -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 启用JupyterLab扩展
RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter --no-build && \
    jupyter labextension install jupyterlab-topbar-extension --no-build && \
    jupyter lab build --dev-build=False --minimize=False

# 配置代码格式化器
RUN jupyter server extension enable --py jupyterlab_code_formatter
