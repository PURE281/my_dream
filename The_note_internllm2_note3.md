# 书生·浦语大模型学习第三节学习笔记
## 学习内容概述
### 学习笔记
1. 介绍了RAG的基本知识内容
RAG（Retrieval Augmented Generation）是一中结合了检索（Retrieval）和生成（Generation）的技术，旨在通过利用外部知识库来增强大型语言模型（LLMs）的性能。它通过检索与用户输入相关的信息片段，并结合这些信息来生成更准确、更丰富的回答
![image](https://github.com/PURE281/my_dream/assets/93171238/52220a9e-ed87-43d4-a398-03311ded5e0e)
2. 工作原理
索引->检索->生成
将外部的知识库分割成chunk，编码成向量，存储在向量数据库（Vector-DB）中->获取到用户的问题后将问题也编码成向量，然后在向量数据库中找到最相关的文档块（top-k chunks）->将检索到的文档块和原始问题一起作为prompt输入到LLM中，生成最终的回答
这个方式在百度千帆大模型中也有使用过，通过在模型中添加外部的知识库（问答对），用户提问的问题若是问答对中出现的，则会匹配到知识库中，然后将知识库中的回答一起作为prompt给到千帆大模型中，然后生成回复
3. 向量数据库（Vector-DB）
- 数据存储
- 相似性检索
- 向量表示的优化
![image](https://github.com/PURE281/my_dream/assets/93171238/9f551f25-b2d2-476d-8810-42dabc9ed2ce)

4. RAG目前的三个类别
![image](https://github.com/PURE281/my_dream/assets/93171238/25a91afe-bd5b-42c1-836d-dd2f0a8fa7e4)
5. RAG常见的优化方法
嵌入优化-结合稀疏和密集检索 多任务
索引优化-细粒度分割（Chunk） 元数据
查询优化-查询扩展、转换 多查询
上下文管理-重排（rerank） 上下文选择/压缩
迭代检索-根据初始查询和迄今为止生成的文本进行重复搜索
递归检索-迭代细化搜索查询 链式推理（Chain-of-Thought）指导检索过程
自适应检索-Flare，Self=RAG 使用LLMs主动决定检索的最佳时机和内容
LLM微调-检索微调 生成微调 双重微调
![image](https://github.com/PURE281/my_dream/assets/93171238/fcd0a700-2f21-4cc9-8933-d872c2c8b4ae)

6. RAG和微调（Fine-tuning）的区别 
RAG
- 非参数记忆，利用外部知识库提供实时更新的信息
- 能够处理知识密集型任务，提供准确的事实性回答
- 通过检索增强，可以生成更多样化的内容
适用场景
- 适用于需要结合最新信息和实时数据的任务；开放域问答、实时新闻摘要等

优势
- 动态知识更新，处理长尾知识问题

局限
- 依赖于外部知识库的质量和覆盖范围。依赖大模型能力

微调 Fine-tuning
- 参数记忆，通过在特定任务数据上的训练，模型可以更好地适应该任务
- 通常需要大量标注数据来进行有效微调
- 微调后的模型可能过拟合，导致泛化能力下降

适用场景
- 适用于数据可用且需要模型高度专业化的任务，如特定领域的文本分类， 情感分享、文本生成等

优势
- 模型性能针对特定任务优化

局限
- 需要大量的标注数据，且对新任务的适应性较差

![image](https://github.com/PURE281/my_dream/assets/93171238/560e6027-e90b-443f-83b4-52f353ff2bf8)
7. 评估框架和基准测试
- 经典评估指标
准确率（Accuracy）、召回率（Recall）、F1分数（F1 Score）、BLEU分数（用于机器翻译和文本生成）、ROUGE分数（用于文本生成的评估）
- RAG 测评框架
基准测试-RGB、RECALL、CRUD
评测工具-RAGAS/ARES/TruLens
![image](https://github.com/PURE281/my_dream/assets/93171238/d2316701-54fb-4f3f-9dd9-10bb96e018f5)
## 实战
### 茴香豆
#### 介绍
一个基于LLMs的领域知识助手，由书生浦语团队开发的开源大模型角。
-  专为即时通讯（IM）工具中的群聊场景优化的工作流，提供及时准确的技术支持和自动化问答服务
-  通过应用检索增强生成（RAG）技术，茴香豆能够理解和搞笑准确的回应与特定知识领域相关的复杂查询
![image](https://github.com/PURE281/my_dream/assets/93171238/0c7d38c4-fd11-45e8-bb4c-af7bbeac567f)
1. 工作流
预处理-》拒绝管道-》响应管道-》返回生成结果
接收到问题后会先经过预处理，然后走拒绝管道，拒绝管道会通过和数据库存储的向量信息和问题进行匹配，获取一个得分，根据得分来判断该问题是否能进入响应管道
![image](https://github.com/PURE281/my_dream/assets/93171238/acbf9ba4-e275-4dd5-86f1-4c592944ed89)
响应管道支持多来源检索：向量数据库，网络搜索结果，知识图谱；生成时也支持混合大模型（本地和远程）；通过安全检查保证生成的内容合规；也通过多重评分拒答工作流避免信息泛滥
![image](https://github.com/PURE281/my_dream/assets/93171238/d0b29c5e-6b53-4163-bc1b-16b1f2e21930)
### 作业1
- 在茴香豆web中创建知识问答助手-进行对话
![image](https://github.com/PURE281/my_dream/assets/93171238/51c6c023-35d7-4615-b3e2-dcced1313bee)
![image](https://github.com/PURE281/my_dream/assets/93171238/f84313f7-72bd-4dac-8759-1093507f23d2)
![image](https://github.com/PURE281/my_dream/assets/93171238/a3771fe8-8a48-4be2-8068-d998e2ab3a8e)
![image](https://github.com/PURE281/my_dream/assets/93171238/60222f29-7a7b-4d3c-ac37-fcf819bcb059)
### 作业2
- 在internstudio搭建茴香豆助手
创建开发机（和上次作业一样不做详细说明）
搭建环境
```
studio-conda -o internlm-base -t InternLM2_Huixiangdou
```
![image](https://github.com/PURE281/my_dream/assets/93171238/4321537c-09cf-40fe-89e9-c1c361a007c8)
查看环境
```
conda env list
```
![image](https://github.com/PURE281/my_dream/assets/93171238/cc7ee0b0-8644-4b6f-8844-c540bdea7a8f)
激活环境
```
conda activate InternLM2_Huixiangdou
```
下载基础文件
```
# 创建模型文件夹
cd /root && mkdir models
# 复制BCE模型
ln -s /root/share/new_models/maidalun1020/bce-embedding-base_v1 /root/models/bce-embedding-base_v1
ln -s /root/share/new_models/maidalun1020/bce-reranker-base_v1 /root/models/bce-reranker-base_v1
# 复制大模型参数（下面的模型，根据作业进度和任务进行**选择一个**就行）
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b
```
下载安装茴香豆
```
# 安装 python 依赖
pip install protobuf==4.25.3 accelerate==0.28.0 aiohttp==3.9.3 auto-gptq==0.7.1 bcembedding==0.1.3 beautifulsoup4==4.8.2 einops==0.7.0 faiss-gpu==1.7.2 langchain==0.1.14 loguru==0.7.2 lxml_html_clean==0.1.0 openai==1.16.1 openpyxl==3.1.2 pandas==2.2.1 pydantic==2.6.4 pymupdf==1.24.1 python-docx==1.1.0 pytoml==0.1.21 readability-lxml==0.8.1 redis==5.0.3 requests==2.31.0 scikit-learn==1.4.1.post1 sentence_transformers==2.2.2 textract==1.6.5 tiktoken==0.6.0 transformers==4.39.3 transformers_stream_generator==0.0.5 unstructured==0.11.2
```
拉取项目
```
cd /root
# 下载 repo
git clone https://github.com/internlm/huixiangdou && cd huixiangdou
git checkout 447c6f7e68a1657fce1c4f7c740ea1700bde0440
```
修改`config.ini`配置文件
![image](https://github.com/PURE281/my_dream/assets/93171238/4f92eba9-124c-490b-92b7-4de3c7e18df3)
创建知识库
  下载 Huixiangdou 语料
  ```
cd /root/huixiangdou && mkdir repodir
git clone https://github.com/internlm/huixiangdou --depth=1 repodir/huixiangdou
  ```
运行下面的命令，增加茴香豆相关的问题到接受问题示例中：
```
cd /root/huixiangdou
mv resource/good_questions.json resource/good_questions_bk.json

echo '[
    "mmpose中怎么调用mmyolo接口",
    "mmpose实现姿态估计后怎么实现行为识别",
    "mmpose执行提取关键点命令不是分为两步吗，一步是目标检测，另一步是关键点提取，我现在目标检测这部分的代码是demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth   现在我想把这个mmdet的checkpoints换位yolo的，那么应该怎么操作",
    "在mmdetection中，如何同时加载两个数据集，两个dataloader",
    "如何将mmdetection2.28.2的retinanet配置文件改为单尺度的呢？",
    "1.MMPose_Tutorial.ipynb、inferencer_demo.py、image_demo.py、bottomup_demo.py、body3d_pose_lifter_demo.py这几个文件和topdown_demo_with_mmdet.py的区别是什么，\n2.我如果要使用mmdet是不是就只能使用topdown_demo_with_mmdet.py文件，",
    "mmpose 测试 map 一直是 0 怎么办？",
    "如何使用mmpose检测人体关键点？",
    "我使用的数据集是labelme标注的，我想知道mmpose的数据集都是什么样式的，全都是单目标的数据集标注，还是里边也有多目标然后进行标注",
    "如何生成openmmpose的c++推理脚本",
    "mmpose",
    "mmpose的目标检测阶段调用的模型，一定要是demo文件夹下的文件吗，有没有其他路径下的文件",
    "mmpose可以实现行为识别吗，如果要实现的话应该怎么做",
    "我在mmyolo的v0.6.0 (15/8/2023)更新日志里看到了他新增了支持基于 MMPose 的 YOLOX-Pose，我现在是不是只需要在mmpose/project/yolox-Pose内做出一些设置就可以，换掉demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py 改用mmyolo来进行目标检测了",
    "mac m1从源码安装的mmpose是x86_64的",
    "想请教一下mmpose有没有提供可以读取外接摄像头，做3d姿态并达到实时的项目呀？",
    "huixiangdou 是什么？",
    "使用科研仪器需要注意什么？",
    "huixiangdou 是什么？",
    "茴香豆 是什么？",
    "茴香豆 能部署到微信吗？",
    "茴香豆 怎么应用到飞书",
    "茴香豆 能部署到微信群吗？",
    "茴香豆 怎么应用到飞书群",
    "huixiangdou 能部署到微信吗？",
    "huixiangdou 怎么应用到飞书",
    "huixiangdou 能部署到微信群吗？",
    "huixiangdou 怎么应用到飞书群",
    "huixiangdou",
    "茴香豆",
    "茴香豆 有哪些应用场景",
    "huixiangdou 有什么用",
    "huixiangdou 的优势有哪些？",
    "茴香豆 已经应用的场景",
    "huixiangdou 已经应用的场景",
    "huixiangdou 怎么安装",
    "茴香豆 怎么安装",
    "茴香豆 最新版本是什么",
    "茴香豆 支持哪些大模型",
    "茴香豆 支持哪些通讯软件",
    "config.ini 文件怎么配置",
    "remote_llm_model 可以填哪些模型?"
]' > /root/huixiangdou/resource/good_questions.json
  ```
再创建一个测试用的问询列表，用来测试拒答流程是否起效：
```
cd /root/huixiangdou

echo '[
"huixiangdou 是什么？",
"你好，介绍下自己"
]' > ./test_queries.json
```
在确定好语料来源后，运行下面的命令，创建 RAG 检索过程中使用的向量数据库：
```
# 创建向量数据库存储目录
cd /root/huixiangdou && mkdir workdir 

# 分别向量化知识语料、接受问题和拒绝问题中后保存到 workdir
python3 -m huixiangdou.service.feature_store --sample ./test_queries.json
```
`tips` 这里可能会报下图异常
![image](https://github.com/PURE281/my_dream/assets/93171238/eaf90129-de75-4fe7-8cb9-998855a3eb9d)
在对应的脚本中增大长度即可
![image](https://github.com/PURE281/my_dream/assets/93171238/627f29e2-4ad5-4b12-bee3-5e39d0aaf03e)
运行茴香豆知识助手
```
# 填入问题
sed -i '74s/.*/    queries = ["huixiangdou 是什么？", "茴香豆怎么部署到微信群", "今天天气怎么样？"]/' /root/huixiangdou/huixiangdou/main.py

# 运行茴香豆
cd /root/huixiangdou/
python3 -m huixiangdou.main --standalone
```
![image](https://github.com/PURE281/my_dream/assets/93171238/0f85f640-1e11-4914-9d6a-f73920619399)
![image](https://github.com/PURE281/my_dream/assets/93171238/0e73d2ed-1c4f-479d-acfc-3c7f923e324e)
![image](https://github.com/PURE281/my_dream/assets/93171238/5ccb918e-af34-41fa-9f89-41c4126ba30e)
![image](https://github.com/PURE281/my_dream/assets/93171238/85a24b48-b225-4f88-9f5d-77fef80dcc21)
![image](https://github.com/PURE281/my_dream/assets/93171238/594f7d15-89cb-46bc-a1b3-b6d800a90160)



