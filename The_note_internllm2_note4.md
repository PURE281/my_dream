# 书生·浦语大模型学习第四节学习笔记
忙半天才发现这是第四节的学习内容（捂脸）公告（两个第五节作业）发的太有迷惑性了
## 学习笔记
1. Finetune简介
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
# 如果你是在 InternStudio 平台，则从本地 clone 一个已有 pytorch 的环境：
# pytorch    2.0.1   py3.10_cuda11.7_cudnn8.5.0_0

studio-conda xtuner0.1.17
# 如果你是在其他平台：
# conda create --name xtuner0.1.17 python=3.10 -y

# 激活环境
conda activate xtuner0.1.17
# 进入家目录 （~的意思是 “当前用户的home路径”）
cd ~
# 创建版本文件夹并进入，以跟随本教程
mkdir -p /root/xtuner0117 && cd /root/xtuner0117

# 拉取 0.1.17 的版本源码
git clone -b v0.1.17  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
# git clone -b v0.1.15 https://gitee.com/Internlm/xtuner

# 进入源码目录
cd /root/xtuner0117/xtuner

# 从源码安装 XTuner
pip install -e '.[all]'
```
执行上面代码，一键搭建conda以及xtuner所需的portch cuda cudnn 最后一步还是一如既往的漫长需等待 
- 数据集准备
由于数据集准备的部分是新建文件和代码，因此在上一步漫长的安装依赖的过程中我手动的进行了文件的创建（大人动作，小孩不要模仿）
一切顺利，不做描述，直接上图
![image](https://github.com/PURE281/my_dream/assets/93171238/57b07b81-e517-4655-ad82-eb5e5dc58860)
![image](https://github.com/PURE281/my_dream/assets/93171238/ad06f2bb-32a5-43f3-89c3-6e2950ab9326)
- 修改完成后运行 generate_data.py 文件
```
# 确保先进入该文件夹
cd /root/ft/data

# 运行代码
python /root/ft/data/generate_data.py
```
脚本运行后同一目录下生成了json格式的文件
![image](https://github.com/PURE281/my_dream/assets/93171238/2d9dfe6f-de49-448d-b6fe-729c1b4c677a)

- 模型准备
```
# 创建目标文件夹，确保它存在。
# -p选项意味着如果上级目录不存在也会一并创建，且如果目标文件夹已存在则不会报错。
mkdir -p /root/ft/model

# 复制内容到目标文件夹。-r选项表示递归复制整个文件夹。
cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b/* /root/ft/model/
```
![image](https://github.com/PURE281/my_dream/assets/93171238/c33f2333-670d-4ba7-9b27-f6589bf1ce6e)

- 配置文件选择
```
# 列出所有内置配置文件
# xtuner list-cfg

# 假如我们想找到 internlm2-1.8b 模型里支持的配置文件
xtuner list-cfg -p internlm2_1_8b
```
![image](https://github.com/PURE281/my_dream/assets/93171238/07729902-4569-4674-b41d-b7b078b45afd)

- 创建配置文件
```
# 创建一个存放 config 文件的文件夹
mkdir -p /root/ft/config

# 使用 XTuner 中的 copy-cfg 功能将 config 文件复制到指定的位置
xtuner copy-cfg internlm2_1_8b_qlora_alpaca_e3 /root/ft/config
```
一切顺利,继续
以上便完成了前期准备,文件目录如下
![image](https://github.com/PURE281/my_dream/assets/93171238/5b368f64-85b1-4700-97a5-4d413ea0dc30)
![image](https://github.com/PURE281/my_dream/assets/93171238/dd7f1505-dcc6-497d-8b5c-20a8a01a0742)
![image](https://github.com/PURE281/my_dream/assets/93171238/e82d61ec-6af6-4cf8-b18e-0c9efb9eaf9e)

- 配置文件修改
进入上面创建的`internlm2_1_8b_qlora_alpaca_e3`配置文件,替换修改后的代码并保存
以下是常用的超参信息
![image](https://github.com/PURE281/my_dream/assets/93171238/ae1ec88d-5919-4c15-9737-725d5eaedcf9)

- 模型训练

-- 指定保存路径
  xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train

-- 模型续训
  xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train --resume /root/ft/train/iter_600.pth

训练开始,又是一个漫长的等待时间...
可以看到训练的过程中返回的一些信息
图一![image](https://github.com/PURE281/my_dream/assets/93171238/10769a1f-f22c-4a21-9b9b-53d7d1d4a630)

图二![image](https://github.com/PURE281/my_dream/assets/93171238/2b9bd1f0-5bf9-40ea-b699-db7dd2dfc0c6)

图二相比图一可以看出回复的内容是有进步的
等待的过程再接着往下看还需要做什么

- 模型转换
模型转换的本质其实就是将原本使用 Pytorch 训练出来的模型权重文件转换为目前通用的 Huggingface 格式文件，那么我们可以通过以下指令来实现一键转换。
```
# 创建一个保存转换后 Huggingface 格式的文件夹
mkdir -p /root/ft/huggingface

# 模型转换
# xtuner convert pth_to_hf ${配置文件地址} ${权重文件地址} ${转换后模型保存地址}
xtuner convert pth_to_hf /root/ft/train/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/ft/train/iter_768.pth /root/ft/huggingface
```
直接上图
![image](https://github.com/PURE281/my_dream/assets/93171238/f1f0711d-ad0a-4805-847a-9c8cc10c8afd)

- 模型整合
```
# 创建一个名为 final_model 的文件夹存储整合后的模型文件
mkdir -p /root/ft/final_model

# 解决一下线程冲突的 Bug 
export MKL_SERVICE_FORCE_INTEL=1

# 进行模型整合
# xtuner convert merge  ${NAME_OR_PATH_TO_LLM} ${NAME_OR_PATH_TO_ADAPTER} ${SAVE_PATH} 
xtuner convert merge /root/ft/model /root/ft/huggingface /root/ft/final_model
```
一切顺利 直接上图 ![image](https://github.com/PURE281/my_dream/assets/93171238/0f0febe2-482e-4e6a-a9e9-36a4f490dda5)

- 对话测试
以上步骤完成就可以进行对话测试了
```
# 与模型进行对话
xtuner chat /root/ft/final_model --prompt-template internlm2_chat
```
测试如下
![image](https://github.com/PURE281/my_dream/assets/93171238/834de4a2-a98f-4a18-a2ce-ab2bbfde3ca7)
嗯？怎么好像我的比视频里训练出来的好一点，好像没有那么过拟合（开心.jpg）
![image](https://github.com/PURE281/my_dream/assets/93171238/0998cc8f-51bd-4cd7-9188-aec2e4378af2)
哦好吧，模型疯掉了
试试没有微调之前的模型，看看它原本又是怎么回复的吧
吐槽-开发机每次执行命令行的时间真的要等好久啊（抓狂.jpg）
嗯 微调前的模型比较顺手
![image](https://github.com/PURE281/my_dream/assets/93171238/b283d89f-b73a-49e0-8d8a-68639519da5b)

使用 --adapter 参数与完整的模型进行对话
```
xtuner chat /root/ft/model --adapter /root/ft/huggingface --prompt-template internlm2_chat
```
- web 部署
拉取前端可视化所需的项目
```
# 创建存放 InternLM 文件的代码
mkdir -p /root/ft/web_demo && cd /root/ft/web_demo

# 拉取 InternLM 源文件
git clone https://github.com/InternLM/InternLM.git

# 进入该库中
cd /root/ft/web_demo/InternLM
```
下载安装所需的库
```pip install streamlit==1.24.0```
将 /root/ft/web_demo/InternLM/chat/web_demo.py 中的内容替换为以下的代码（与源代码相比，此处修改了模型路径和分词器路径，并且也删除了 avatar 及 system_prompt 部分的内容，同时与 cli 中的超参数进行了对齐）。
```
"""This script refers to the dialogue example of streamlit, the interactive
generation code of chatglm2 and transformers.

We mainly modified part of the code logic to adapt to the
generation of our model.
Please refer to these links below for more information:
    1. streamlit chat example:
        https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    2. chatglm2:
        https://github.com/THUDM/ChatGLM2-6B
    3. transformers:
        https://github.com/huggingface/transformers
Please run with the command `streamlit run path/to/web_demo.py
    --server.address=0.0.0.0 --server.port 7860`.
Using `python path/to/web_demo.py` may cause unknown problems.
"""
# isort: skip_file
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers.utils import logging

from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip

logger = logging.get_logger(__name__)


@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 2048
    top_p: float = 0.75
    temperature: float = 0.1
    do_sample: bool = True
    repetition_penalty: float = 1.000


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
            input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None \
        else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None \
        else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            break


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model():
    model = (AutoModelForCausalLM.from_pretrained('/root/ft/final_model',
                                                  trust_remote_code=True).to(
                                                      torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained('/root/ft/final_model',
                                              trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=32768,
                               value=2048)
        top_p = st.slider('Top P', 0.0, 1.0, 0.75, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.1, step=0.01)
        st.button('Clear Chat History', on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    return generation_config


user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'


def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = ('')
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def main():
    # torch.cuda.empty_cache()
    print('load model begin.')
    model, tokenizer = load_model()
    print('load model end.')


    st.title('InternLM2-Chat-1.8B')

    generation_config = prepare_generation_config()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('What is up?'):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
        })

        with st.chat_message('robot'):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,  # pylint: disable=undefined-loop-variable
        })
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
```
最后是老生常谈的端口映射
```
# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 开发机分配的端口号
```
输入密码然后访问localhost:6006即可
又是一段漫长的等待时间....
页面正常显示后输入进行对话
![image](https://github.com/PURE281/my_dream/assets/93171238/42b27e19-8d35-4d93-ab06-4e802eefc164)
微调的数据可以回答，但其他就开始答非所问了...
![image](https://github.com/PURE281/my_dream/assets/93171238/01d014bf-73ff-4345-aabf-967d0f6a1772)


## 进阶作业

优秀学员必做
### 作业1
将自我认知的模型上传到 OpenXLab，并将应用部署到 OpenXLab
### 作业2
复现多模态微调
在基础作业的项目环境下，运行虚拟环境
拉取项目
```
cd ~ && git clone https://github.com/InternLM/tutorial -b camp2 && conda activate xtuner0.1.17 && cd tutorial

python /root/tutorial/xtuner/llava/llava_data/repeat.py \
  -i /root/tutorial/xtuner/llava/llava_data/unique_data.json \
  -o /root/tutorial/xtuner/llava/llava_data/repeated_data.json \
  -n 200
```
准备配置文件
```
cp /root/tutorial/xtuner/llava/llava_data/internlm2_chat_1_8b_llava_tutorial_fool_config.py /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py

```
创建配置文件
```
# 查询xtuner内置配置文件
xtuner list-cfg -p llava_internlm2_chat_1_8b

# 拷贝配置文件到当前目录
xtuner copy-cfg \
  llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune \
  /root/tutorial/xtuner/llava
```
修改配置文件
修改llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py文件中的：
·pretrained_pth
·llm_name_or_path
·visual_encoder_name_or_path
·data_root
·data_path
·image_folder
```
# Model
- llm_name_or_path = 'internlm/internlm2-chat-1_8b'
+ llm_name_or_path = '/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b'
- visual_encoder_name_or_path = 'openai/clip-vit-large-patch14-336'
+ visual_encoder_name_or_path = '/root/share/new_models/openai/clip-vit-large-patch14-336'

# Specify the pretrained pth
- pretrained_pth = './work_dirs/llava_internlm2_chat_1_8b_clip_vit_large_p14_336_e1_gpu8_pretrain/iter_2181.pth'  # noqa: E501
+ pretrained_pth = '/root/share/new_models/xtuner/iter_2181.pth'

# Data
- data_root = './data/llava_data/'
+ data_root = '/root/tutorial/xtuner/llava/llava_data/'
- data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
+ data_path = data_root + 'repeated_data.json'
- image_folder = data_root + 'llava_images'
+ image_folder = data_root

# Scheduler & Optimizer
- batch_size = 16  # per_device
+ batch_size = 1  # per_device


# evaluation_inputs
- evaluation_inputs = ['请描述一下这张图片','Please describe this picture']
+ evaluation_inputs = ['Please describe this picture','What is the equipment in the image?']

```
开始Finetune
```
cd /root/tutorial/xtuner/llava/
xtuner train /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py --deepspeed deepspeed_zero2
```
Finetune后
加载 1.8B 和 Fintune阶段产物 到显存。
```
# 解决小bug
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

# pth转huggingface
xtuner convert pth_to_hf \
  /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py \
  /root/tutorial/xtuner/llava/work_dirs/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy/iter_1200.pth \
  /root/tutorial/xtuner/llava/llava_data/iter_1200_hf

# 启动！
xtuner chat /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
  --visual-encoder /root/share/new_models/openai/clip-vit-large-patch14-336 \
  --llava /root/tutorial/xtuner/llava/llava_data/iter_1200_hf \
  --prompt-template internlm2_chat \
  --image /root/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
```
![image](https://github.com/PURE281/my_dream/assets/93171238/df7bcb49-6f92-4288-b40a-7701f6f38403)
结束
