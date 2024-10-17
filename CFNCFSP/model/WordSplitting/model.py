import os
import pickle
from typing import List, Union
from ltp import LTP as PLTP
import torch

from CFNCFSP.data.WordSpliting_dicts import POSmap
from CFNCFSP.schema.base import SentenceResult, TargetParsing, Word, Sentence, Target
from CFNCFSP.schema.frame import POS
from CFNCFSP.utils import cfn_cache


class WSModel():
    """_summary_
    """
    name = "WS"

    def __init__(
            self,
            device: Union[str, torch.device] = "cpu",
            *args,
            **kwargs
        ):

        ws_pretrained_model_name_or_path = kwargs.get("ws_pretrained_model_name_or_path", None)

        if ws_pretrained_model_name_or_path is None:
            ws_pretrained_model_name_or_path = 'LTP/base'
        self.ltp_model = PLTP(pretrained_model_name_or_path=ws_pretrained_model_name_or_path).to(device)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_file = os.path.abspath(os.path.join(script_dir, '../../data/all_targets.bin'))
        #targets_set={('target1','pos'),('target2','pos')......}
        with open(target_file, 'rb') as f:
            self.targets_set = pickle.load(f)  


    @cfn_cache.cache()
    def predict(self, sentences: List[str], targets) -> List[SentenceResult]:
        try:
            result = self.ltp_model.pipeline(sentences, tasks = ["cws","pos"])
            cws = result.cws
            pos = result.pos

            inputs = [
                SentenceResult(
                    sentence = Sentence(
                        text = s,
                        words = [
                            Word( 
                                word = w,
                                pos = POS(
                                    POS_id = POSmap.mean2index.get(p, "112"),
                                    POS_name = POSmap.mean2word.get(p, "CFN专有词元"),
                                    POS_mark = p
                                )
                            )
                            for w, p in zip(w_s, p_s)
                        ]
                    )
                )
                for s, (w_s, p_s) in zip(sentences, zip(cws, pos))
            ]

            for one_input in inputs:
                begin = 0
                one_input.parsing = []
                for content in one_input.sentence.words:
                    if (content.word, content.pos.POS_mark) in self.targets_set:
                        start = begin
                        end = start + len(content.word) - 1

                        one_input.parsing.append(
                            TargetParsing(
                                target = Target(
                                    start = start,
                                    end = end,
                                    word = content.word,
                                    pos = content.pos
                                )
                            )
                        )

                    begin += len(content.word)

            return inputs            
        except :
            raise  "查找目标词错误"
    
    
    def cut(self, inputs: List[SentenceResult], targets: List[List[int]]):
        if not targets:
            return inputs

        tmp_parsing = [[] for _ in range(len(inputs))]
        for t in targets:
            idx = t[0]
            p_start = t[1]
            p_end = t[2]
            for x in inputs[idx].parsing:
                if p_start == x.target.start and p_end == x.target.end:
                    x.target.frame = None
                    x.target.frame_with_score = None
                    x.arguments = None
                    tmp_parsing[idx].append(x)
                    break
        
        for i in range(len(inputs)):
            inputs[i].parsing = tmp_parsing[i]
        
        return inputs 
