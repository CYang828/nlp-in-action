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