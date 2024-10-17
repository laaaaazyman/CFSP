import torch
import torch.nn.functional as F
from typing import List

from CFNCFSP.model.base import CFNBaseModel
from CFNCFSP.schema.base import SentenceResult
from CFNCFSP.schema.frame import FeResult, FeWithScore
from CFNCFSP.data.RoleIdentification_dict import Rolemap

class RIModel(CFNBaseModel):
    name = "RI"

    def _predict(self, inputs: List[SentenceResult], targets: List[List[int]], *args, **kwargs) -> List[SentenceResult]:
        """_summary_
            模型预测
        Returns:
            ...: _description_
        """
        n = kwargs.get("n", 10)
        parsing_objs = kwargs.get("parsing_obj")

        arguments_objs = []
        for idx, p in enumerate(parsing_objs):
            arguments_objs.extend([idx, argument] for argument in p.arguments)

        arguments = []
        for one_argument in arguments_objs:
            if one_argument[1].end + 1 < targets[one_argument[0]][1]:
                arguments.append([one_argument[0], one_argument[1].start + 1, one_argument[1].end + 1])
            elif one_argument[1].start + 1 > targets[one_argument[0]][2]:
                arguments.append([one_argument[0], one_argument[1].start + 3, one_argument[1].end + 3])

        if len(arguments) == 0:
            return inputs

        processed_data = self.tokenizer.batch_encode_plus([one_sentenceresult.sentence.text for one_sentenceresult in inputs], padding = True)
        input_ids = processed_data.data["input_ids"]
        attention_mask = processed_data.data["attention_mask"]

        processed_input_ids = []
        processed_attention_mask = []
        for target in targets:
            processed_input_ids.append(input_ids[target[0]][0: target[1]] + [1] + input_ids[target[0]][target[1]: target[2] + 1] + [2] + input_ids[target[0]][target[2] + 1:])
            processed_attention_mask.append(attention_mask[target[0]][0: target[1]] + [1] + attention_mask[target[0]][target[1]: target[2] + 1] + [1] + attention_mask[target[0]][target[2] + 1:])

        input_ids = torch.tensor(processed_input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(processed_attention_mask, dtype=torch.long).to(self.device)

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_token = (bert_out["hidden_states"][-4] + bert_out["hidden_states"][-3] + bert_out["hidden_states"][-2] + bert_out["hidden_states"][-1]) / 4
        logits = self.compute_model_result(hidden_token)
        arguments = torch.tensor(arguments)
        al = logits[arguments[:,0],:,arguments[:,1],arguments[:,2]]
        probs = F.softmax(al, dim=-1)
        topk_scores, topk_indices = torch.topk(probs, k=n, dim=-1)
        topk_scores, topk_indices = topk_scores.tolist(), topk_indices.tolist()

        for s, i, arg in zip(topk_scores, topk_indices, arguments_objs):
            arg[1].fe = FeResult(                           
                    fe_id = i[0],
                    fe_name = self.id2label[i[0]],
                    fe_abbr = Rolemap.index2fe_abbr[i[0]]
                )
            arg[1].fe_with_score = [
                FeWithScore(
                    fe_id = fe_id,
                    fe_name = self.id2label[fe_id],
                    fe_abbr = Rolemap.index2fe_abbr[fe_id],
                    score = fe_score
                )
                for fe_id, fe_score in zip(i[1:], s[1:])
            ]

        return inputs









