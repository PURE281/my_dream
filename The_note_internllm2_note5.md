# 书生·浦语大模型学习第五节学习笔记
## 笔记
1. 大模型部署面临的挑战
- 计算量巨大
![image](https://github.com/PURE281/my_dream/assets/93171238/73f83374-5bae-496b-85b4-6a5e22dfb42a)
- 内存开销巨大
![image](https://github.com/PURE281/my_dream/assets/93171238/ebdc9db9-c7da-41ba-a4ba-9a91a0c421d4)
- 访存瓶颈
这个比较有意思
显卡的计算能力是足够的，但是访存能力十分有限，因此导致了显卡在数据交换上花费了大量的时间，只有极少的时间在进行实际的计算上
虽然可以通过batch-size的设置增加访存量，但是性价比太低
![image](https://github.com/PURE281/my_dream/assets/93171238/afc14864-be28-4db7-9aed-ac180aa1e95d)
- 优化方案
-- 模型剪枝
  移除模型中不必要或多余的组件，比如参数，让模型更加搞笑。通过对模型中贡献有限的冗余参数进行剪枝，在保证性能最低下降的同时，减少存储需求， 提高计算效率
  - 非结构化剪枝
    移除个别参数，不考虑整体网络结构。将低于阈值的参数置零的方式对个别权重或神经元进行处理
  - 结构化剪枝
    根据预定义规则移除链接或分层结构，同时保持整体网络结构。一次性地针对整组权重，优势在于降低模型复杂性和内存使用，同时保证整体的LLM结构完整
  ![image](https://github.com/PURE281/my_dream/assets/93171238/7ebfc819-c9aa-488e-8f20-c3489ada3eeb)

-- 知识蒸馏
  通过引导轻量化的学生模型“模仿”性能更好，结构更复杂的教师模型，在不改变学生模型结构的情况下提高性能
  上下文学习，思维链，指令跟随
  ![image](https://github.com/PURE281/my_dream/assets/93171238/baba9d15-21f4-41e8-be18-08d05fa00694)

-- 量化
  量化技术将传统的表示方法中的浮点数转换为证书或其他离散形式，以减轻深度学习模型的存储和计算负担
  量化感知训练、量化感知微调、训练后量化
  知识点：由于调用大模型的过程是一种访存密集型的过程，因此量化的过程虽然增加了量化和反量化两个计算过程，但通过量化的操作减轻了内存负担，最终还是实现了提高性能的目的。也就是说虽然增加了计算负担，但由于硬件上的计算能力是过剩的，因此增加的计算对整体的性能影响可以忽略不计；但是对访存而言，减轻的内存带来的性能提升是很大的，因此综合下来，量化的方式是能有效提升性能的
![image](https://github.com/PURE281/my_dream/assets/93171238/ba7a83a4-64ca-44aa-aa05-c7dc80b5ae7f)

2. LMDeploy部署
LMDeploy简介
涵盖了LLM任务的全套轻量化、部署和服务解决方案。核心功能包括高效推理、可靠量化、便捷服务和有状态推理
![image](https://github.com/PURE281/my_dream/assets/93171238/740e048c-0ea1-43ab-a987-092422462d1b)

## 作业
### 基础作业
1. 配置 LMDeploy 运行环境
```
studio-conda -t lmdeploy -o pytorch-2.1.2

conda activate lmdeploy

pip install lmdeploy[all]==0.3.0
```
InternStudio开发机上下载模型（推荐）--软链接的方式
```
cd ~
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
# cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
```
2. 以命令行方式与 InternLM2-Chat-1.8B 模型对话
![image](https://github.com/PURE281/my_dream/assets/93171238/3113c4eb-2531-4ba1-ba78-24882691fc1a)

2.1 使用Transformer库运行模型
   在root下新建一个`pipeline_transformer.py`文件
   复制代码
```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/internlm2-chat-1_8b", trust_remote_code=True)

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("/root/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

inp = "hello"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=[])
print("[OUTPUT]", response)

inp = "please provide three suggestions about time management"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=history)
print("[OUTPUT]", response)

```
然后运行 当前时间17:56分
![image](https://github.com/PURE281/my_dream/assets/93171238/30e2fe42-4a55-4255-ab2d-106db450d8fe)

等待至17:58分
![image](https://github.com/PURE281/my_dream/assets/93171238/2688f2f2-1082-4bf9-bf4c-8e682e51ab81)

2.2 使用LMDeploy与模型对话
当前时间17:59
![image](https://github.com/PURE281/my_dream/assets/93171238/069256eb-0514-4efb-97f5-f6830640b925)
等待至18:01分
![image](https://github.com/PURE281/my_dream/assets/93171238/7b68dd34-17a9-4b3a-a4d3-e3db2a5e903c)
体感上来说，LMDeploy加载的时间比Transfromer快一点

2.3 LMDeploy模型量化(lite)
主要包括 KV8量化和W4A16量化。总的来说，量化是一种以参数或计算中间结果精度下降换空间节省（以及同时带来的性能提升）的策略。
- 计算密集（compute-bound）: 指推理过程中，绝大部分时间消耗在数值计算上；针对计算密集型场景，可以通过使用更快的硬件计算单元来提升计算速度。
- 访存密集（memory-bound）: 指推理过程中，绝大部分时间消耗在数据读取上；针对访存密集型场景，一般通过减少访存次数、提高计算访存比或降低访存量来优化。
常见的 LLM 模型由于 Decoder Only 架构的特性，实际推理时大多数的时间都消耗在了逐 Token 生成阶段（Decoding 阶段），是典型的访存密集型场景。
KV8量化是指将逐 Token（Decoding）生成过程中的上下文 K 和 V 中间结果进行 INT8 量化（计算时再反量化），以降低生成过程中的显存占用。W4A16 量化，将 FP16 的模型权重量化为 INT4，Kernel 计算时，访存量直接降为 FP16 模型的 1/4，大幅降低了访存成本。Weight Only 是指仅量化权重，数值计算依然采用 FP16（需要将 INT4 权重反量化）。

### 进阶作业
由于已过ddl，待后面有时间再完成...
![image](https://github.com/PURE281/my_dream/assets/93171238/e3ffe843-46f9-42e8-b972-723dd2b0d821)
