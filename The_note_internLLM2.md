# 书生·浦语大模型学习第一节学习笔记
## 契机
契机是去年的棒球大赛上（bushi），在B站上刷到了复旦大学做的一个关于心理咨询小助手的智能对话模型，实际体验了一下发现是很接近我想象中的心理咨询模式的智能助手，并且项目也开源了，视频里说了项目就是以interllm为基础，通过添加数据集训练和微调后进行开发的，让我很受鼓舞，因此前来学习
## 学习内容概述
1. 由于第一节视频以介绍为主，个人认为实际且有用的知识点不多，因此不做过多知识总结
2. 视频里介绍了大模型有很多功能，比如长文本分析，内生计算，代码解释等，这些目前并不是太关注，因为之前试过让gpt等大模型进行此类功能时体验和使用并不友好，因此只能说未来可期，还有很大的进步空间；个人比较关注“文科类”的功能，比如智能对话，看视频说“文科类”的能力甚至超过了gpt4，尤其是中文方面，很期待，因为希望能通过数据训练部署一个专属的心理咨询，情绪疏导的智能助手
3. internLM2有几个主要的亮点
![image](https://github.com/PURE281/my_dream/assets/93171238/05a305d2-44c5-4508-9da0-41d0ae653600)
超长上下文（20万token；在理科能力上的提升如推理，数学，代码；对话和创作体验；也支持工具多轮调用，复杂智能体搭建以及加入代码解释后的数理能力和数据分析能力
4. 
![image](https://github.com/PURE281/my_dream/assets/93171238/72c4a35a-9593-491f-b4a1-3406cc9b618b)
5. 书生浦语全链条开源开放体系
如下图，数据集，预训练，微调，部署，评测以及应用
![image](https://github.com/PURE281/my_dream/assets/93171238/107b08e4-543d-44f8-aa44-98c093145b83)
6. 在opencompass榜单上，书生浦语的internLM2也紧跟在很多闭源的大厂大模型身后，甚至在某些能力上有所超越
![image](https://github.com/PURE281/my_dream/assets/93171238/7ad4e5eb-d14e-47ba-a223-4759cda7aaff)
7. 也支持LMDeloy部署
接口支持python，gRPC，RESTful；也支持openai-server，gradio，triton inference server服务以及4bit权重和8bit k/v的轻量化处理
![image](https://github.com/PURE281/my_dream/assets/93171238/7f6f0bbc-9988-41f2-8d43-f494330bc013)
8. 智能体
支持多种类型的智能体能力如ReAct,ReWoo,AutoGPT；也支持多种大语音模型，如GPT-3.5/4 internLM，Hugging Face Transformers 以及Llama；也能拓展和支持丰富的工具，如文生图，文生语音，图片描述；搜索，计算器，代码解释器以及Rapid API等
![image](https://github.com/PURE281/my_dream/assets/93171238/07d2d074-3a38-4ac5-8fa3-203095176bd0)
## 心得
https://www.bilibili.com/video/BV1Vx421X72D
1. 视频看下来，大概讲述了人工智能模型发展趋势以及书生浦语的历程。
2. 访问了compass rank，看到书生浦语这个开源项目的排名居然紧随文心一言，通义千问这类的闭源大模型，并且7b的数据集就可以和通义千问的14b相媲美，能力确实强大。
3. interllm2也支持量化部署，8g就可以跑起来，很期待在本地部署属于自己的智能助手
4. 其他的由于刚开始接触大模型，还处于一知半解的程度，多模态，deploy之类的还不是很懂
--20240330 9am
--20240404 10pm
