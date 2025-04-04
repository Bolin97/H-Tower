# HTOWER对话数据集生成与分析系统

## 项目简介
本项目是一个用于生成和分析对话数据集的系统，主要包含三个核心功能模块：
1. **对话数据生成**：通过HTOWER方法(agent2agent.py)和常规方法(simplified_dialogue.py)生成模拟对话数据
2. **数据集切分**：根据不同用户类型(普通、期望型、指数型)对对话数据进行切分
3. **指标计算**：计算对话质量相关指标，评估对话效果

系统支持多个领域的对话生成，包括书籍、汽车、房产、写字楼和视频等。

## 目录结构
```
web8/
├── data/                   # 数据集目录
│   ├── book/               # 书籍领域数据
│   │   ├── book.json       # 领域基础数据
│   │   ├── book_token_tf.json # 特征权重数据
│   │   ├── book_tree.json  # 领域特征树结构
│   │   └── user_profile.json # 用户画像数据
│   ├── car/                # 汽车领域数据(结构同上)
│   ├── house/              # 房产领域数据(结构同上)
│   ├── office_building/    # 写字楼领域数据(结构同上)
│   └── video/              # 视频领域数据(结构同上)
├── agent2agent.py          # HTOWER方法对话生成主模块
├── simplified_dialogue.py  # 常规方法对话生成主模块
├── gpt_api.py              # API调用接口和提示词处理
├── computer.py             # 指标计算模块
├── token_select_strategy.py # 特征选择策略
├── user_simulator.py       # 用户模拟器
├── llm_Infer.py            # 大模型推理模块
├── default_data.py         # 默认数据和常量定义
├── prompt.py               # 提示词模板
├── agent_data_process.py   # 数据处理工具
├── agent_gpt.py            # Agent行为定义
├── cal_average_length.py   # 长度计算工具
├── webpage.py              # 网页数据处理
│
├── expectation_cut.py      # 期望用户数据集切分
├── exponential_cut.py      # 指数用户数据集切分
├── round_cut.py            # 普通用户数据集切分
├── expectation_cut_htower.py # HTOWER期望用户切分
├── exponential_cut_htower.py # HTOWER指数用户切分
└── round_cut_htower.py     # HTOWER普通用户切分
```

## 环境要求
- Python 3.8+
- 依赖的Python包见下方安装说明

## 依赖安装
项目依赖可以通过以下命令安装：
```bash
pip install openai requests tqdm pypinyin torch transformers python-Levenshtein
```

或者创建一个requirements.txt文件，内容如下：
```
openai>=1.0.0
requests>=2.25.0
tqdm>=4.62.0
pypinyin>=0.46.0
torch>=1.9.0
transformers>=4.11.0
python-Levenshtein>=0.12.2
```

然后执行：
```bash
pip install -r requirements.txt
```

## API密钥配置

### 1. 常规方法API配置
在`simplified_dialogue.py`中配置以下API密钥和URL：
```python
# agent对话特征提取大模型api
api_extract_agent = 'your_api_key_here'
url_extract_agent = 'your_api_url_here'
model_extract_agent='your_model_name_here'

# user对话特征提取大模型api
api_extract_user = 'your_api_key_here'
url_extract_user = 'your_api_url_here'
model_extract_user='your_model_name_here'

# agent对话大模型api
api_agent = 'your_api_key_here'
url_agent = 'your_api_url_here'
model_agent='your_model_name_here'

# user对话大模型api
api_user = 'your_api_key_here'
url_user = 'your_api_url_here'
model_user='your_model_name_here'
```

### 2. HTOWER方法API配置
在`gpt_api.py`文件的387-404行配置HTower方法所需的API密钥：
```python
def call_gpt(question, gpt_type, model="gpt-3.5-turbo"):
    # 在这里配置您的OpenAI API密钥
    client = OpenAI(api_key="your_api_key_here")
    
    # 或者如果使用自定义API端点
    # client = OpenAI(base_url="your_api_url_here", api_key="your_api_key_here")
    
    # 配置模型名称
    # model = "your_model_name_here"
    
    # 其余代码...
```

## 运行方法

### 1. 对话数据生成

#### HTOWER方法
```bash
python agent2agent.py
```

可选参数：
- `--domain`: 指定对话领域，可选值：book, car, house, office_building, video
- `--user_type`: 用户类型，可选值：normal, expectation, exponential
- `--max_rounds`: 最大对话轮数，默认为10

用户类型参数配置：
1. 指数型用户(exponential)参数：
   - `token_long`: token长度阈值，默认值150
   - `k`: Logistic函数斜率参数，默认值2.55
   - `x0`: Logistic函数中点参数，默认值1.55

2. 期望型用户(expectation)参数：
   - `alpha`: 收益参数(0 < alpha < 1)，默认值0.88
   - `beta`: 损失参数(beta < 1)，默认值0.88
   - `lambda_`: 损失厌恶系数(lambda_ > 1)，默认值2.25

示例：
```bash
# 使用默认参数
python agent2agent.py --domain car --user_type expectation --max_rounds 15

# 自定义用户参数
python agent2agent.py --domain car --user_type exponential --max_rounds 15 --token_long 200 --k 2.8 --x0 1.6
python agent2agent.py --domain car --user_type expectation --max_rounds 15 --alpha 0.85 --beta 0.85 --lambda_ 2.5
```

#### 常规方法
```bash
python simplified_dialogue.py
```

可选参数：
- `--user_type`: 用户类型，可选值：normal, exponential, expectation，默认为normal
- `--max_rounds`: 最大对话轮数，默认为10

示例：
```bash
python simplified_dialogue.py --user_type exponential --max_rounds 12
```

### 2. 数据集切分

#### 常规方法切分
```bash
python expectation_cut.py input.json output.json 阈值
python exponential_cut.py input.json output.json 阈值
python round_cut.py input.json output.json 阈值
```

#### HTOWER方法切分
```bash
python expectation_cut_htower.py input.json output.json 阈值
python exponential_cut_htower.py input.json output.json 阈值
python round_cut_htower.py input.json output.json 阈值
```

参数说明：
- `input.json`: 输入对话数据文件路径
- `output.json`: 输出结果文件路径
- `阈值`: 用于控制对话截断的敏感度(0-1之间的小数)，值越小截断越早

示例：
```bash
python round_cut.py data/conversations.json data/round_cut_result.json 0.5
```

### 3. 指标计算
```bash
python computer.py input.json
```

参数说明：
- `input.json`: 需要计算指标的数据文件路径

示例：
```bash
python computer.py data/round_cut_result.json
```

## 输出结果说明

### 对话生成输出
对话生成会在项目根目录下生成conversations.json文件，包含完整的对话记录和需求匹配信息。

### 数据集切分输出
切分操作会根据指定的输出文件名生成切分后的数据集，包含截断后的对话和相关指标。

### 指标计算输出
指标计算会输出以下几个关键指标：
- 平均信息完整度：agent获取的需求与用户原始需求的匹配程度
- 平均Features命中率：agent询问的特征在用户需求中的命中比例
- 平均集合相似度：使用Levenshtein距离和BERT模型计算的相似度指标
