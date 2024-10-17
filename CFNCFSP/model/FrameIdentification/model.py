import torch
import torch.nn.functional as F
from typing import List

from CFNCFSP.model.base import CFNBaseModel
from CFNCFSP.schema.base import SentenceResult
from CFNCFSP.schema.frame import FrameResult, FrameWithScore

class FIModel(CFNBaseModel):
    name = "FI"
    
    def cut(self, inputs: List[SentenceResult], targets: List[List[int]]):
        inputs = super().cut(inputs, targets)
        for x in inputs:
            for t in x.parsing:
                t.arguments = None
        return inputs


    def _predict(self, inputs: List[SentenceResult], targets: List[List[int]], *args, **kwargs) -> List[SentenceResult]:
        """
        n:输出推荐的框架的个数
            _summary_
            模型预测
        Returns:
            ...: _description_
        """
        n = kwargs.get("n", 10)
        parsing_objs = kwargs.get("parsing_obj")

        processed_data = self.tokenizer.batch_encode_plus([one_sentenceresult.sentence.text for one_sentenceresult in inputs], padding = True)
        input_ids = input_ids = processed_data.data["input_ids"]
        attention_mask = processed_data.data["attention_mask"]
        
        target_objs = [p.target for p in parsing_objs]

        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_token = (bert_out["hidden_states"][-4] + bert_out["hidden_states"][-3] + bert_out["hidden_states"][-2] + bert_out["hidden_states"][-1]) / 4
        logits = self.compute_model_result(hidden_token)
        targets = torch.tensor(targets).to(self.device)
        al = logits[targets[:, 0], :, targets[:, 1], targets[:, 2]] 
        probs = F.softmax(al, dim=-1)
        topk_scores, topk_indices = torch.topk(probs, k=n, dim=-1) 
        topk_scores, topk_indices = topk_scores.tolist(), topk_indices.tolist()  

        for s, i, tgt in zip(topk_scores, topk_indices, target_objs):
            tgt.frame = FrameResult(                           
                    frame_id = i[0],
                    frame_name = self.id2label[i[0]]
                )
            tgt.frame_with_score = [
                FrameWithScore(
                    frame_id = f_id,
                    frame_name = self.id2label[f_id],
                    score = f_score
                )
                for f_id, f_score in zip(i[1:], s[1:])
            ]

        return inputs


