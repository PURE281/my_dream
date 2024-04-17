# 书生·浦语大模型学习第二节学习笔记

## 学习笔记
1. Finetune简
- 两种Finetune范式
增量预训练微调
- 使用场景：让基座模型学习到一些新知识，如某个垂直领域的常识
- 训练数据：文章，书籍，代码等
指令跟随微调
- 使用场景：让模型学会对话模板，根据人类指令进行对话
- 训练数据：高质量的对话、对答数据
流程：InternLM基座模型-》增量预训练-》InterLM垂类基座模型-》指令跟随-》InterLM垂类对话模型
![image](https://github.com/PURE281/my_dream/assets/93171238/59d1d38a-bcfe-4e2a-a228-66f54b6024a7)
![image](https://github.com/PURE281/my_dream/assets/93171238/0c5e8f64-2300-4567-a5b0-08c4edf0fe7b)
- 一条数据的一生
数据训练模板，训练格式以及训练结果
![image](https://github.com/PURE281/my_dream/assets/93171238/9799432f-3812-42f9-b281-cab64d2cd9a4)
标准化格式数据
x-tuner会将我们输入的json格式的数据一键转换成InterLM2支持的数据格式，再根据这个格式进行训练
![image](https://github.com/PURE281/my_dream/assets/93171238/0fdb3fe4-833a-4caa-a0d7-77c6cbac3526)
LoRA原理介绍
在基座模型上训练一个小模型，保证最大程度满足需求的同时减少了显存开销
![image](https://github.com/PURE281/my_dream/assets/93171238/944e59e6-b475-4bda-82da-87fc3e34fc32)
QLoRA
LORA的mini版
全参数微调，LoRA微调及QLoRA微调比较
![image](https://github.com/PURE281/my_dream/assets/93171238/c4c867a1-af77-41d7-9532-a5be8a151408)

2. X-Tune介绍
傻瓜式及轻量式
7b可以在消费级显卡上跑，确实很吸引人进行尝试（3070请求一战）
![image](https://github.com/PURE281/my_dream/assets/93171238/94e83e84-4ada-451d-807b-87a7b5cf6615)
3. InternLM2-1.8b
InternLM2-1.8b、InternLM2-Chat-1.8b-SFT、InternLM2-Chat-1.8b
![image](https://github.com/PURE281/my_dream/assets/93171238/e4c77c29-fab8-4a31-bc51-f45d4205bb36)

4. 多模态LLM
原本的LLM是只支持输入文本，然后通过文本Embadding模型转换成向量，再进行文本生成的过程
多模态LLM可以在支持文本输入的同时也支持图像输入，并且通过图像处理模型（Image Projector）将图像转换成向量，最后生成文本
![image](https://github.com/PURE281/my_dream/assets/93171238/682ea031-3ec0-4876-91f2-865cfde0fd92)
6. LLoRA
目前可以粗略的将文本单模型LLM+Image Projector统称为LLoRA模型
![image](https://github.com/PURE281/my_dream/assets/93171238/b02136f0-8bfc-4d6a-b528-e4aaa9d2bdc7)

## 基础作业
作业要求：训练自己的小助手认知（记录复现过程并截图）
- 开发机准备
前几个笔记皆有记录，不做描述
由于上一个作业茴香豆问题较多，为保证内存足够，将茴香豆项目及对应的conda环境删除（绝对不是因为进阶作业没完成而闹情绪）
![image](https://github.com/PURE281/my_dream/assets/93171238/f6e77f20-6e7b-4bfc-9846-c86feb32f730)
删除conda环境执行```conda remove -n InternLM2-Huixiangdou --all```代码即可，剩下的茴香豆项目手动进行文件删除即可
![image](https://github.com/PURE281/my_dream/assets/93171238/b92cf09e-0df3-40d1-8c3b-e431f900fee0)
等待了一些时间，不过还是删除成功了（爽！）
- i创建环境
```
studio-conda xtuner0.1.17
```
执行上面代码，一键搭建conda以及xtuner所需的portch cuda cudnn

## 进阶作业

优秀学员必做
### 作业1
将自我认知的模型上传到 OpenXLab，并将应用部署到 OpenXLab
### 作业2
复现多模态微调
