## [解决]git-ssh: connect to host github.com port 22: Connection timed out
1. 先测试可用性
ssh -T -p 443 git@ssh.github.com

2.然后编辑 ~/.ssh/config 文件，如果没有config文件的话就直接 vim ~/.ssh/config加入以下内容
Host github.com
Hostname ssh.github.com
Port 443

3.再次测试
ssh -T git@github.com
提示如下就说明成功了

## gensim function() argument 'code' must be code, not str

```bash
pip install --upgrade aiohttp
```

[issue地址](https://github.com/aio-libs/aiohttp/issues/6794)

## ModuleNotFoundError: No module named ‘altair.vegalite.v4’

```bash
pip install "altair<5"
```
https://discuss.streamlit.io/t/modulenotfounderror-no-module-named-altair-vegalite-v4/42921/6


## 
