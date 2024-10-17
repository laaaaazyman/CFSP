from functools import wraps
from typing import Any
import warnings
from CFNCFSP.schema.base import SentenceResult
from copy import deepcopy


class Cache():

    def __init__(self) -> None:
        self.__cache_table = {} # 句子: 对应的那张二维表格
        self.__task2id = {
            'FI': 1, 
            'AI': 2, 
            'RI': 3 
        }        

    def __call__(self) -> Any:
        def _fun(func):
            @wraps(func)
            def _func_in(*args, **kwargs):
                # 取出对应的数据
                model = args[0]
                task = model.name
                n = kwargs.get('n')
                inputs = kwargs.get('inputs', args[1] if len(args) > 1 else [])
                if 'targets' in kwargs:
                    targets = kwargs['targets']
                # if not kwargs.get('targets') and len(args) == 3:
                #     targets = args[2]
                # else:
                #     targets = []
                elif len(args) > 2:
                    targets = args[2]
                else:
                    targets = []

                no_ws_cache_k = []  # 此次inputs中没有WS的句子
                tables = [0] * len(inputs)  # 此次任务所需要用到的，即没有执行过这个任务的
                for id, i in enumerate(inputs):
                    if isinstance(i, SentenceResult):
                        cache_k = i.sentence.text
                    elif isinstance(i, str):
                        cache_k = i
                    else:
                        raise '您输入的inputs不符合要求'
                    table = self.__cache_table.get(cache_k, None)
                    if not table:
                        no_ws_cache_k.append(cache_k) # 所有待做WS的句子
                    else:
                        tables[id] = table
                # 只有inputs中有无法从缓存中查到的sentence时候，才会走这里
                if no_ws_cache_k:
                    ws_res = func(model, no_ws_cache_k, targets)
                    for r in ws_res:
                        self.__cache_table[r.sentence.text] = [r, {}]
                        for p_idx, t in enumerate(r.parsing):
                            self.__cache_table[r.sentence.text][1][(t.target.start, t.target.end)] = [0, p_idx]
                        # 将其塞入它应该在的地方， 即与他本来在inputs的位置相同
                        tables[tables.index(0)] = self.__cache_table[r.sentence.text]


                r_targets, r_inputs, r_parsing = self.subtract(tables, targets, task) # 获取要做的targets 从二维表中
                if r_inputs: # 没有这个变量说明，在缓存中有你要的内容，取出cut返回即可
                    _ = func(model, r_inputs, r_targets, parsing_obj = r_parsing, n = n)
                    self.plus(r_inputs, r_targets, task) # 把新做的加进去二维表中
                res = [table[0] for table in tables] # 此处是全取出, 后面cut再筛
                sres = deepcopy(res)
                sres = model.cut(sres, targets)
                return sres
            return _func_in
        return _fun


    def plus(self, r_inputs, r_targets, task):
        progress = self.__task2id.get(task, 0)
        for t in r_targets:
            self.__cache_table[r_inputs[t[0]].sentence.text][1][(t[1] - 1, t[2] - 1)][0] = progress


    def subtract(self, tables, targets, task):
        progress = self.__task2id.get(task, 0)
        r_targets = []
        r_inputs = []
        r_parsing = []
        if not targets: #没targets，就找此任务中所有<progress的targets去做后续任务
            for sentence_result, table in tables:
                flag = False
                for t, v in table.items():
                    if v[0] < progress:
                        r_targets.append([len(r_inputs), t[0] + 1, t[1] + 1])
                        r_parsing.append(sentence_result.parsing[v[1]])
                        flag = True
                if flag:
                    r_inputs.append(sentence_result)

        else: #有给定的targets，所以按照那个选取
            added_sentence_idx = set()
            idx = -1
            for sentence_idx, t_start, t_end in targets:
                flag = tables[sentence_idx][1].get((t_start, t_end))
                if flag is None:
                    warnings.warn(f'The character that the index:({t_start}, {t_end}) refers to is not a target word', UserWarning)
                    continue
                if flag[0] < progress:
                    if sentence_idx not in added_sentence_idx:
                        added_sentence_idx.add(sentence_idx)
                        idx += 1
                    r_targets.append([idx, t_start + 1, t_end + 1])
                    if not tables[sentence_idx][0] in r_inputs:
                        r_inputs.append(tables[sentence_idx][0])
                    r_parsing.append(tables[sentence_idx][0].parsing[flag[1]])        
            
        return  r_targets, r_inputs, r_parsing


    def clear(self):
        self.__cache_table = {}


def log_decorator(func):
    def wrapper(self, *args, **kwargs):
        print(f"Calling function: {func.__name__}")
        return func(self, *args, **kwargs)
    return wrapper

cache = Cache()