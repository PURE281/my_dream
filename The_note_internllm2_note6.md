![image](https://github.com/PURE281/my_dream/assets/93171238/f0318b65-385c-4ccf-a9d6-aafe52a00dfb)# 书生·浦语大模型学习第六节学习笔记

## 学习笔记

1. 概述
1.0 智能体
- 包括三个部分
大脑：作为控制器，承担记忆、思考和决策任务。接收来自感知模块的信息并采取相应动作
感知：对外部环境的多模态信息进行感知和处理。包括但不限于图像、音频、视频、传感器等
动作：利用并执行工具以影响环境。工具可能包括文本的检索、调用相关API、操控机械臂等。
![image](https://github.com/PURE281/my_dream/assets/93171238/40a3ad4f-183b-45ba-a332-ef3debaab9de)
- 智能体范式
AutoGPT、ReWoo、ReAct
![image](https://github.com/PURE281/my_dream/assets/93171238/39bf067d-99ce-4287-a33e-78b3bdc61323)

1.1 Lagent 是什么
Lagent 是一个轻量级开源智能体框架，旨在让用户可以高效地构建基于大语言模型的智能体。同时它也提供了一些典型工具以增强大语言模型的能力。

Lagent 目前已经支持了包括 AutoGPT、ReAct 等在内的多个经典智能体范式，也支持了如下工具：
Arxiv 搜索
Bing 地图
Google 学术搜索
Google 搜索
交互式 IPython 解释器
IPython 解释器
PPT
Python 解释器
1.2 AgentLego 是什么
AgentLego 是一个提供了多种开源工具 API 的多模态工具包，旨在像是乐高积木一样，让用户可以快速简便地拓展自定义工具，从而组装出自己的智能体。通过 AgentLego 算法库，不仅可以直接使用多种工具，也可以利用这些工具，在相关智能体框架（如 Lagent，Transformers Agent 等）的帮助下，快速构建可以增强大语言模型能力的智能体。
![image](https://github.com/PURE281/my_dream/assets/93171238/d03713c6-3867-435f-bafc-426f7860cfe0)

1.3 两者的关系
经过上面的介绍，我们可以发现，Lagent 是一个智能体框架，而 AgentLego 与大模型智能体并不直接相关，而是作为工具包，在相关智能体的功能支持模块发挥作用。

两者之间的关系可以用下图来表示：
flowchart LR
    subgraph Lagent
        tool[调用工具]
        subgraph AgentLego
            tool_support[工具功能支持]
        end
        tool_output(工具输出)
        tool --> tool_support --> tool_output
    end

    input(输入) --> LLM[大语言模型]
    LLM --> IF{是否需要调用工具}
    IF -->|否| output(一般输出)
    IF -->|是| tool
    tool_output -->|处理| agent_output(智能体输出)

## 作业
### 基础作业
1. 完成 Lagent Web Demo 使用，并在作业中上传截图
1.1 创建开发机
   配置为30%的A100 conda12.2
1.2 创建环境
   ```
   mkdir -p /root/agent
   studio-conda -t agent -o pytorch-2.1.2
   ```
   等待一段时间
   ![image](https://github.com/PURE281/my_dream/assets/93171238/aaa56923-6ee7-42e9-9639-93f8c1e6e519)
1.3 安装 Lagent 和 AgentLego
   ```
   cd /root/agent
   conda activate agent
   git clone https://gitee.com/internlm/lagent.git
   cd lagent && git checkout 581d9fb && pip install -e . && cd ..
   git clone https://gitee.com/internlm/agentlego.git
   cd agentlego && git checkout 7769e0d && pip install -e . && cd ..
   ```
   等待一段时间
![image](https://github.com/PURE281/my_dream/assets/93171238/4e56debe-cacf-4142-89c4-fa728188dc39)

1.4 安装其他依赖
  ```
  conda activate agent
  pip install lmdeploy==0.3.0
  ```
![image](https://github.com/PURE281/my_dream/assets/93171238/98f8081b-c5ea-4526-9315-a13c2db21e08)

1.5 准备 Tutorial
  ```
  cd /root/agent
  git clone -b camp2 https://gitee.com/internlm/Tutorial.git
  ```

1.6 使用 LMDeploy 部署
  由于 Lagent 的 Web Demo 需要用到 LMDeploy 所启动的 api_server，因此我们首先按照下图指示在 vscode terminal 中执行如下代码使用 LMDeploy 启动一个 api_server。
  ```
conda activate agent
lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b \
                            --server-name 127.0.0.1 \
                            --model-name internlm2-chat-7b \
                            --cache-max-entry-count 0.1
```
![image](https://github.com/PURE281/my_dream/assets/93171238/ebbe7c06-2dee-4e3d-8f9a-5407dc3b8511)

1.7 启动并使用 Lagent Web Demo
![image](https://github.com/PURE281/my_dream/assets/93171238/6dd346d2-6761-4a86-80e0-5a6dea44cc66)
在本地进行端口映射，将 LMDeploy api_server 的23333端口以及 Lagent Web Demo 的7860端口映射到本地
```
ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 你的 ssh 端口号
```
![image](https://github.com/PURE281/my_dream/assets/93171238/c50ca04f-1638-44de-acb3-4a6c1caf35f0)

接下来在本地的浏览器页面中打开 http://localhost:7860 以使用 Lagent Web Demo。首先输入模型 IP 为 127.0.0.1:23333，在输入完成后按下回车键以确认。并选择插件为 ArxivSearch，以让模型获得在 arxiv 上搜索论文的能力。
tips:输入的`127.0.0.1:23333`前面不能留空格，否则会报异常，tz最好也不要挂...
![image](https://github.com/PURE281/my_dream/assets/93171238/4890a125-52ab-4e94-ac1b-8ab5e4aad31b)
将上面的异常解决后回复正常
![image](https://github.com/PURE281/my_dream/assets/93171238/d9e6500b-3055-4230-8067-7cd075b75435)

1.8 用 Lagent 自定义工具
使用 Lagent 自定义工具主要分为以下几步：
- 继承 BaseAction 类
- 实现简单工具的 run 方法；或者实现工具包内每个子工具的功能
- 简单工具的 run 方法可选被 tool_api 装饰；工具包内每个子工具的功能都需要被 tool_api 装饰

下面我们将实现一个调用和风天气 API 的工具以完成实时天气查询的功能。

1.8.1 创建工具文件
首先通过 touch /root/agent/lagent/lagent/actions/weather.py（大小写敏感）新建工具文件，该文件内容如下：
```
import json
import os
import requests
from typing import Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

class WeatherQuery(BaseAction):
    """Weather plugin for querying weather information."""
    
    def __init__(self,
                 key: Optional[str] = None,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)
        key = os.environ.get('WEATHER_API_KEY', key)
        if key is None:
            raise ValueError(
                'Please set Weather API key either in the environment '
                'as WEATHER_API_KEY or pass it as `key`')
        self.key = key
        self.location_query_url = 'https://geoapi.qweather.com/v2/city/lookup'
        self.weather_query_url = 'https://devapi.qweather.com/v7/weather/now'

    @tool_api
    def run(self, query: str) -> ActionReturn:
        """一个天气查询API。可以根据城市名查询天气信息。
        
        Args:
            query (:class:`str`): The city name to query.
        """
        tool_return = ActionReturn(type=self.name)
        status_code, response = self._search(query)
        if status_code == -1:
            tool_return.errmsg = response
            tool_return.state = ActionStatusCode.HTTP_ERROR
        elif status_code == 200:
            parsed_res = self._parse_results(response)
            tool_return.result = [dict(type='text', content=str(parsed_res))]
            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = str(status_code)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return
    
    def _parse_results(self, results: dict) -> str:
        """Parse the weather results from QWeather API.
        
        Args:
            results (dict): The weather content from QWeather API
                in json format.
        
        Returns:
            str: The parsed weather results.
        """
        now = results['now']
        data = [
            f'数据观测时间: {now["obsTime"]}',
            f'温度: {now["temp"]}°C',
            f'体感温度: {now["feelsLike"]}°C',
            f'天气: {now["text"]}',
            f'风向: {now["windDir"]}，角度为 {now["wind360"]}°',
            f'风力等级: {now["windScale"]}，风速为 {now["windSpeed"]} km/h',
            f'相对湿度: {now["humidity"]}',
            f'当前小时累计降水量: {now["precip"]} mm',
            f'大气压强: {now["pressure"]} 百帕',
            f'能见度: {now["vis"]} km',
        ]
        return '\n'.join(data)

    def _search(self, query: str):
        # get city_code
        try:
            city_code_response = requests.get(
                self.location_query_url,
                params={'key': self.key, 'location': query}
            )
        except Exception as e:
            return -1, str(e)
        if city_code_response.status_code != 200:
            return city_code_response.status_code, city_code_response.json()
        city_code_response = city_code_response.json()
        if len(city_code_response['location']) == 0:
            return -1, '未查询到城市'
        city_code = city_code_response['location'][0]['id']
        # get weather
        try:
            weather_response = requests.get(
                self.weather_query_url,
                params={'key': self.key, 'location': city_code}
            )
        except Exception as e:
            return -1, str(e)
        return weather_response.status_code, weather_response.json()
```

1.8.2 在和风天气创建项目获取key
![image](https://github.com/PURE281/my_dream/assets/93171238/0fdcfeeb-a094-426c-8cc0-748c47db26c1)

再次调用 LMDeploy 服务以及 Web Demo 服务
![image](https://github.com/PURE281/my_dream/assets/93171238/0a918421-cfe5-4285-af27-9df1fa9e5f44)

在本地执行端口映射 同上不做赘述
输入127.0.0.1::23333 选取天气插件 输入天气问题 回答如下
![image](https://github.com/PURE281/my_dream/assets/93171238/8eaa82b1-829c-4b61-842e-07d870f94030)

2. 完成 AgentLego 直接使用部分，并在作业中上传截图

2.1 直接使用 AgentLego
2.1.1 下载demo文件
```
cd /root/agent
wget http://download.openmmlab.com/agentlego/road.jpg
```
2.1.2 安装依赖
```
conda activate agent
pip install openmim==0.3.9
mim install mmdet==3.3.0
```
2.1.3 在 /root/agent 目录下新建 direct_use.py 以直接使用目标检测工具
```
import re

import cv2
from agentlego.apis import load_tool

# load tool
tool = load_tool('ObjectDetection', device='cuda')

# apply tool
visualization = tool('/root/agent/road.jpg')
print(visualization)

# visualize
image = cv2.imread('/root/agent/road.jpg')

preds = visualization.split('\n')
pattern = r'(\w+) \((\d+), (\d+), (\d+), (\d+)\), score (\d+)'

for pred in preds:
    name, x1, y1, x2, y2, score = re.match(pattern, pred).groups()
    x1, y1, x2, y2, score = int(x1), int(y1), int(x2), int(y2), int(score)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(image, f'{name} {score}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

cv2.imwrite('/root/agent/road_detection_direct.jpg', image)
```
然后执行刚才创建的py文件
```
python /root/agent/direct_use.py
```
![image](https://github.com/PURE281/my_dream/assets/93171238/f81b8f92-a8d9-4fdf-889b-6078bf60fea1)

![image](https://github.com/PURE281/my_dream/assets/93171238/5b1e3f19-9224-4ef0-90f2-de6ad0e4784f)

2.2 作为智能体工具使用
2.2.1 修改相关文件
由于 AgentLego 算法库默认使用 InternLM2-Chat-20B 模型，因此我们首先需要修改 /root/agent/agentlego/webui/modules/agents/lagent_agent.py 文件的第 105行位置，将 internlm2-chat-20b 修改为 internlm2-chat-7b

![image](https://github.com/PURE281/my_dream/assets/93171238/98ceebe2-a0b9-4f6f-8d6f-ff1525df191d)
2.2.2 使用 LMDeploy 部署
部署流程同上,先启动LMDEploy所需的api_server 然后再运行启动 AgentLego WebUI 最后由于需要访问web因此需要进行端口映射
```
conda activate agent
lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b \
                            --server-name 127.0.0.1 \
                            --model-name internlm2-chat-7b \
                            --cache-max-entry-count 0.1
```
```
conda activate agent
cd /root/agent/agentlego/webui
python one_click.py
```
```
ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 你的 ssh 端口号
```
2.2.3 使用 AgentLego WebUI
点击上方 Agent 进入 Agent 配置页面。（如①所示）
点击 Agent 下方框，选择 New Agent。（如②所示）
选择 Agent Class 为 lagent.InternLM2Agent。（如③所示）
输入模型 URL 为 http://127.0.0.1:23333 。（如④所示）
输入 Agent name，自定义即可，图中输入了 internlm2。（如⑤所示）
点击 save to 以保存配置，这样在下次使用时只需在第2步时选择 Agent 为 internlm2 后点击 load 以加载就可以了。（如⑥所示）
点击 load 以加载配置。（如⑦所示）

![image](https://github.com/PURE281/my_dream/assets/93171238/e1902174-d0d8-4ed9-8581-5f8820597156)

然后配置工具，如下图所示。
- 点击上方 Tools 页面进入工具配置页面。（如①所示）
- 点击 Tools 下方框，选择 New Tool 以加载新工具。（如②所示）
- 选择 Tool Class 为 ObjectDetection。（如③所示）
- 点击 save 以保存配置。（如④所示）

结果
[Uploading image.png…]()
嗯结果是出来了，但是为啥返回的是英文的....
