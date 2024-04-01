# 书生·浦语大模型学习第二节学习笔记
## 学习内容概述
### 学习笔记
根据第二节课的文档打开开发机，照着文档内容执行命令，运行脚本
大致和平时的项目相同
1. 创建python环境
2. 通过git下载需要运行的项目代码
3. 通过py脚本运行项目所需的模型资源
4. 通过命令行运行项目

新学的内容是可以通过本地ssh和服务器（internstudio）连接的方式在本地通过localhost的方式访问服务器上的项目 不过具体是怎么实现的还不太清楚

### 实践1 部署 InternLM2-Chat-1.8B 模型进行智能对话
1. 在`internstudio`创建开发机
![image](https://github.com/PURE281/my_dream/assets/93171238/606ae254-9de7-442c-b07c-47f3b5575272)
3. 根据文档执行相关命令 创建模型运行所需环境
4. 运行初demo
![image](https://github.com/PURE281/my_dream/assets/93171238/0fd4391e-0c6e-4483-8a10-576beec57c45)

### 实践2 部署实战营优秀作品 八戒-Chat-1.8B 模型
1. 在上一个作业中点击exit 退出demo
2. 通过命令重新激活demo conda activate demo
3. 通过git 下载八戒项目 然后通过命令行下载八戒项目所需模型
4. 通过命令行运行项目
![image](https://github.com/PURE281/my_dream/assets/93171238/02853302-56ca-4087-8f8d-66508cef9118)
6. 本地电脑通过powershell 输入命令行 链接项目
7. 通过浏览器输入localhost:127.0.0.1:6006 等待响应即可
8. 开启对话
![image](https://github.com/PURE281/my_dream/assets/93171238/2c531c26-e3f3-4ac6-8a6a-726ed76a5bf0)


遇到的两个问题
1. 通过 `streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006` 运行提示异常
![image](https://github.com/PURE281/my_dream/assets/93171238/4b0e4813-c151-4228-82f2-4b49f24422d9)
原因可能是先在本地powershell和interstudio进行了链接，将本地powershell关闭后重新输入上方命令行运行成功
2. 通过本地powershell和internstudio进行链接，提示permission denied 字样
原因：密码粘贴时出了问题，可通过ctrl+insert 或 ctrl+v输入，然后直接回车即可

### 实践3 熟悉huggingface下载功能 使用 `huggingface_hub` python包,下载`InternLm-Chat-7b`的`config.json`文件到本地


### 实践3 通过 InternLM2-Chat-7B 运行 Lagent 智能体 Demo
![image](https://github.com/PURE281/my_dream/assets/93171238/0ecb9ac1-fa02-42dd-a7d0-376622d325c1)

### 实践4 实践部署 浦语·灵笔2 模型
#### 图文创作
#### 视觉问答
