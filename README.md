# CFSP

# 仓库已迁移
**此项目仓库已不再维护。新的项目仓库是 [https://github.com/SXUNLP/CFSP](https://github.com/SXUNLP/CFSP)。请访问新存储库以获取更新.**


使用我们的工具包CFSP来分析您输入的多条语句，CFSP将分词与目标词搜索、框架识别、论元范围识别和论元角色识别这些功能链接在一个pipeline函数中，用户根据自身需要指定任务，程序将顺序执行前置任务及相应任务，以json结构返回结果。

## 一.安装

此项目的所有功能在Python 3.7与3.8进行了测试。 CFSP可以通过pip直接下载：

```shell
pip install CFNCFSP
```

或者选择通过源代码直接安装：

```shell
git clone https://github.com/laaaaazyman/CFSP.git
cd CFSP
pip install .
```

## 二.模型及下载地址

|                   分词模型                   | 框架识别模型 | 论元范围识别模型 | 论元角色识别模型 |
| :------------------------------------------: | :----------: | :--------------: | :--------------: |
| [🤗LTP/Base](https://huggingface.co/LTP/base) | 🤗[SXUCFN/CFNParser-FI](https://huggingface.co/SXUCFN/CFNParser-FI) | 🤗[SXUCFN/CFNParser-AI](https://huggingface.co/SXUCFN/CFNParser-AI) |         🤗[SXUCFN/CFNParser-RI](https://huggingface.co/SXUCFN/CFNParser-RI)         |

## 三.快速开始

### 提供模型路径，创建后即可使用

```python
from CFNCFSP import CFNParser
# Step 1 : 创建CFSP对象
# 默认 huggingface 下载，可能需要代理
tool = CFNParser() #使用默认参数，等效于下方代码
tool = CFNParser(	 ws_pretrained_model_name_or_pat = 'LTP/base', 
		    	 fi_pretrained_model_name_or_path = '...',
		       	 ai_pretrained_model_name_or_path = '...',
 			 ri_pretrained_model_name_or_path = '...',
		    	 device = 'cpu',
		    	 n = 10, # n即返回最有可能的n个结果，默认为10
)

# 给出模型下载或解压后的路径
tool = CFNParser(	 ws_pretrained_model_name_or_pat = path/to/ltpbase, 
		    	 fi_pretrained_model_name_or_path = path/to/fi,
		    	 ai_pretrained_model_name_or_path = path/to/ai,
		    	 ri_pretrained_model_name_or_path = path/to/ri,
)

# Step 2 : 指定输入句子，可以是str或者List[str]或者前置任务返回的结果
sentences = [
			'根据建设部的规定，凡属于国际金融组织贷款并由国际公开招标的工程全部由外国投资或赠款建设的工程，以及国内企业在技术上难以单独承包的中外合资建设工程，境外建筑企业在取得中国审批的外国企业承包工程资质证后，皆可进入中国境内承包建设目。',
			'到去年底，全区各项存款余额达七十一点六三亿元，比上年同期增长百分之四十一点七八，其中，城乡居民储蓄存款为十九点三七亿元，比上年同期增长百分之四十八点二。',
    			'今天是个好日子，心想的事儿都能成。'
]


# Step 3 : 执行拟定任务
# 分词并且找目标词WS、框架识别FI、论元范围识别AI和论元角色识别RI(tasks函数不指定时，等同于tasks = ['RI']，等同于顺序执行'WS', 'FI', 'AI', 'RI'所有任务)
# 当不指定targets参数时，等同于寻找所有目标词
res = tool.pipeline(sentences, tasks = ['RI'])

# Step 4 : 以json格式输出
for i in res:
	print(i)
```

### 结果样例：

```python
{
    "sentence": (json)句子信息--由text和words组成一个元素
    {
        "text": (str)原句子,
        "words": (List[json])ltp分词结果的结构化表示
		[
        	{
        		"word": (str)词,
        		"pos": (json)词的词性信息
					{
               			"POS_id": (int)词性id,
                   		"POS_name": (str)词性名称,
                   		"POS_mark": (str)词性标识符号
               		}
			}
		]
    }
    "parsing": (List(json))框架信息--由target和arguments组成一个元素
	[
        {
            "target": (json)目标词信息
            {
                "word": (str)目标词,
                "pos": (json)目标词的词性信息
            	{
                    "POS_id": (int)词性id,
                    "POS_name": (str)词性名称,
                    "POS_mark": (str)词性标识符号
                },
                "start": (int)目标词起始位置(索引从0开始),
                "end": (int)目标词结束位置(索引从0开始),
                "frame": (json)模型输出得分最高的框架信息:
        		{
                    "frame_id": (int)框架id,
                    "frame_name": (str)框架名称
                },
                "frame_with_scores": (List(json))剩余n - 1个可能匹配的框架的信息
				[
                    {
                        "frame_id": (int)框架id,
                        "frame_name": (str)框架名称,
                        "score": (float)模型对应得分
                    },
					{
                        ...
                    },
                    ...
                ]
            },
            "arguments": (List(json))框架对应论元信息--由start/end/fe/fe_with_score组成一个元素
			[
                {
                    "start": (int)论元起始位置(索引从0开始),
                    "end": (int)论元结束位置(索引从0开始),
                    "fe":{
                        "fe_id": (int)论元角色id,
                        "fe_name": (str)论元角色名称,
                        "fe_abbr": (str)论元角色缩写
                    },
                    "fe_with_score": (List(json))剩余n - 1个可能匹配的论元角色的信息
					[
                        {
                            "fe_id": (int)论元角色id,
                            "fe_name": (str)论元角色名称,
                            "fe_abbr": (str)论元角色缩写,
                            "score": (float)模型对应得分
                        },
						{
                            ...
                        },
						...                            
                    ]
                }
            ]
        }
}
```

## 四.pipeline函数参数targets使用说明

### 1.不指定 ： 默认找寻句子中全部目标词进行后续操作

```python
res = tool.pipeline(sentences, tasks = ['RI'])
# except print : 
# 第一句的 '属于'(10-11); '建设'(41-42); '建设'(68-69); '取得'(80-81); '审批'(84-85); '进入'(102-103); '建设'(110-111) 作目标词展开后续任务
# 第二句的 '达'(13-13); '增长'(28-29); '为'(51-51); '增长'(65-66) 作目标词展开后续任务
# 第三句的 '是'(2-2); '成'(15-15) 作目标词展开后续任务
```

### 2.targets使用单个字符进行指定 ： 所有句子都查找此目标词

```python
tar = '建设'
res = tool.pipeline(sentences, tasks = ['RI'],targets = tar)
# except print : 
# 第一句的 '建设'(41-42); '建设'(68-69); '建设'(110-111) 作目标词展开后续任务
# 第二句无对应目标词(即parsing为[])
# 第三句无对应目标词(即parsing为[])
```

### 3.targets使用与sentences等长字符列表进行指定 ： 元素一一对应进行查找

```python
tar = [['建设', '属于'], ['存款', '达'], '成']
res = tool.pipeline(sentences, tasks = ['RI'],targets = tar)
# except print : 
# 第一句的 '属于'(10-11); '建设'(41-42); '建设'(68-69); '建设'(110-111) 作目标词展开后续任务
# 第二句的 '达'(13-13) 作目标词展开后续任务('存款'不在目标词文档中，不查找)
# 第三句的 '成'(15-15) 作目标词展开后续任务
```

### 4.targets使用索引进行指定 ：

```python
tar = [[[2, 3], [10, 11]], [13, 13], [2, 2]]
res = tool.pipeline(sentences, tasks = ['RI'],targets = tar)
# except print : 
# 第一句的 (10-11)：属于，作目标词展开后续任务
# 第二句的 (13-13) ：达，作目标词展开后续任务
# 第三句的 (2-2) ：是，作目标词展开后续任务
```

其余情况如给定句子数与指定targets列表长度不等长，将导致程序报错无法运行。
