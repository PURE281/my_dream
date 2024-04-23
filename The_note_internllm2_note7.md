# 书生·浦语大模型学习第六节学习笔记
## 学习笔记
### OpenCompass大模型测评
用于对训练好的大模型进行测评打分，检验模型能力的工具
### 大模型测评中的挑战
全面性、测评标准、数据污染、鲁棒性
![image](https://github.com/PURE281/my_dream/assets/93171238/04161069-bfe2-4a1f-8ddc-25196f911515)
### 司南大模型测评体系
### 模型评测标准
模型分类：基座模型、对话模型、公开权重的开源模型、API模型
![image](https://github.com/PURE281/my_dream/assets/93171238/ea689b72-9b10-4ef9-ab00-c009624be469)
客观评测与主观测评
客观问答题、客观选择题以及开放式主观问答
![image](https://github.com/PURE281/my_dream/assets/93171238/e5732748-1314-4331-ade9-39b7b85d346f)
提示词工程
明确性、概念无歧义、逐步引导、具体描述、迭代反馈
![image](https://github.com/PURE281/my_dream/assets/93171238/e1993489-f712-4fda-b2d0-ebcee2747f61)
长文本评测
大海捞针式，对长文本内容的识别和处理能力
![image](https://github.com/PURE281/my_dream/assets/93171238/f72eeafd-ac1d-4237-bfa5-84a3fb9543fb)
### opencampass2.0能力升级
基础能力：语言、知识、理解、数学、代码、推理
综合能力：考试、对话、创作、智能体、评价、长文本
![image](https://github.com/PURE281/my_dream/assets/93171238/8e0e09dd-6ece-4eda-9700-63939d04427a)

## 基础作业
使用 OpenCompass 评测 internlm2-chat-1_8b 模型在 C-Eval 数据集上的性能
### 搭建环境
开发机 
cuda 11.7 10%A100
python环境
```
studio-conda -o internlm-base -t opencompass
source activate opencompass
git clone -b 0.2.4 https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

亲测`pip install -e .`安装提示成功了,但是还是无法运行`python tools/list_configs.py internlm ceval`指令
![image](https://github.com/PURE281/my_dream/assets/93171238/2625f6bf-ac9b-4665-bff5-dbddbb02f3d4)

因此使用原始的`pip install -r requirements.txt`
### 数据集准备
复制数据集至opencompass文件夹中并解压
```
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip
```
### 查看支持的数据集和模型
```
python tools/list_configs.py internlm ceval
```
![image](https://github.com/PURE281/my_dream/assets/93171238/df42aac2-9fd7-43f1-8286-fd8cfc57b414)
### 启动评测 (10% A100 8GB 资源)
```
python run.py --datasets ceval_gen --hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 1024 --max-out-len 16 --batch-size 2 --num-gpus 1 --debug
```
异常
![image](https://github.com/PURE281/my_dream/assets/93171238/e6308fa0-4a94-4b8f-a723-791564771478)

解决方案`pip install protobuf` `export MKL_SERVICE_FORCE_INTEL=1`
执行上面命令后再次执行启动评测的代码
![image](https://github.com/PURE281/my_dream/assets/93171238/038e2db0-3197-4153-9bcf-4d8c25498e5e)

![image](https://github.com/PURE281/my_dream/assets/93171238/d23074e6-3711-47e7-bcf5-1b37db925b39)
