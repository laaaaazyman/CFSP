from typing import List, Union
import torch
from transformers import BertTokenizer
from CFNCFSP.model import MODEL_MAP
from CFNCFSP.model.WordSplitting.model import WSModel
from CFNCFSP.model.base import CFNBaseModel
from CFNCFSP.schema.base import SentenceResult
import os
import re

from CFNCFSP.utils.get_dimensions import get_dimensions

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CFNParser():
    def __init__(
            self, 
            ws_pretrained_model_name_or_path: str = 'LTP/base',
            fi_pretrained_model_name_or_path: str = 'SXUCFN/CFNParser-FI',
            ai_pretrained_model_name_or_path: str = 'SXUCFN/CFNParser-AI',
            ri_pretrained_model_name_or_path: str = 'SXUCFN/CFNParser-RI',
            device: Union[str, torch.device] = 'cpu',
            n: int = 10,
        ) -> None:
        """
        先只加载个分词模型即可，后面的用再加
        """
        self.model_name2path = {
            "WS": ws_pretrained_model_name_or_path,
            "FI": fi_pretrained_model_name_or_path,
            "AI": ai_pretrained_model_name_or_path,
            "RI": ri_pretrained_model_name_or_path,
        }
        self.supported_tasks=set(MODEL_MAP.keys())
        ls_model = WSModel(ws_pretrained_model_name_or_path=ws_pretrained_model_name_or_path, device=device)
        self.device = device
        self.n = n
        self.models = {"WS": ls_model}
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name2path.get("FI"))

    def __get_model(self, task: str) -> CFNBaseModel:
        model = self.models.get(task, None)
        if not model:
            model =  MODEL_MAP[task].from_pretrained(self.model_name2path.get(task), self.tokenizer)
            model = model.to(self.device)
            self.models[task] = model
        return model


    @torch.no_grad()
    def pipeline(
            self, 
            inputs: Union[str, List[str], SentenceResult, List[SentenceResult]],
            targets: Union[str, List[str], List[List[str]], List[int], List[List[int]], List[List[List[int]]]] = [],
            tasks: List[str] = ['WS', 'FI', 'AI', 'RI'],
        ) -> List[SentenceResult]:
        all_tasks = list(MODEL_MAP.keys())
        tasks_map = {x:i for i,x in enumerate(all_tasks)}
        result_tasks = []

        #验证
        if not self.supported_tasks.issuperset(tasks):
            raise ValueError(f"Unsupported tasks: {tasks}")

        if isinstance(inputs, str) or isinstance(inputs, SentenceResult):
            inputs = [inputs]

        if targets:
            if isinstance(inputs, List) and isinstance(targets, List):
                if isinstance(targets[0], int):
                    if len(inputs) != 1:
                        raise '输入待处理句子数与targets数不匹配'
                else:
                    if len(inputs) != len(targets):
                        raise '输入待处理句子数与targets数不匹配'

        #? 开始处理targets
        # List[str] -> List[List[str]]
        if targets:
            if isinstance(targets, List) and all(isinstance(target, str) for target in targets):
                targets = [[target] for target in targets]

            # str -> List[List[str]]
            if isinstance(targets, str):
                targets = [[targets] for _ in range(len(inputs))]

            #将targets从List[List[str]] -> List[List[List[int]]]
            flag = 0
            for target in targets:
                if isinstance(target, str):
                    target = [target]
                    flag = 1
                elif isinstance(target, List) and len(target):
                    if isinstance(target[0], str):
                        flag = 1

            if flag:
                tmp = [[] for _ in range(len(targets))]
                for i, target_list in enumerate(targets):
                    for target in target_list:
                        indexes = [[match.start(), match.end() - 1] for match in re.finditer(target, inputs[i])]
                        tmp[i].extend(indexes)
                targets = tmp

            #todo 把前两种int的情况也变为List[List[List[int]]]
            if get_dimensions(targets) == 1:
                targets = [[targets]]
            elif get_dimensions(targets) == 2:
                targets = [[target_idx] for target_idx in targets]

            # todo 改变一下targets结构，从List[List[List[int]]]变为List[List[int]]然后带上表示这是第几句的目标词的i
            tmp = []
            for i, target in enumerate(targets):
                if get_dimensions(target) == 1:
                    if not target:
                        continue
                    start = target[0]
                    end = target[1]
                    tmp.append([i, start, end])
                if get_dimensions(target) == 2:
                    for t in target:
                        start = t[0]
                        end = t[1]
                        tmp.append([i, start, end])
            targets = tmp
        
        if not targets and isinstance(inputs, List) and all(isinstance(i, SentenceResult) for i in inputs):
            targets = ([[idx, p.target.start, p.target.end] for idx, i in enumerate(inputs) for p in inputs[idx].parsing])

        """
            补齐 前置任务
        """
        #这个得尤其注意一下给定inputs是str的情况下是否有bug
        maxnum = max(tasks_map.get(task, 0) for task in tasks)
        result_tasks = [all_tasks[i] for i in range(maxnum + 1)]

        """
            依次执行 模型代码
        """
        output = [i.replace(' ','') for i in inputs]
        for task in result_tasks:
            model = self.__get_model(task)
            output : SentenceResult = model.predict(output, targets, n = self.n)
            if task == 'WS' and targets: 
                # 从output中取肯定正确的targets们
                real_tgt = set([ (i, p.target.start, p.target.end) for i, sentence_result in enumerate(output) for p in sentence_result.parsing ])
                targets = [tuple(t) for t in targets]
                targets = list(real_tgt & set(targets))
                targets = sorted(targets, key=lambda x: (x[0], x[1], x[2]))
                if not targets:
                    return output

        return output
