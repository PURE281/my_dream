# 书生·浦语大模型学习第二节学习笔记
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
### 茴香豆-搭建