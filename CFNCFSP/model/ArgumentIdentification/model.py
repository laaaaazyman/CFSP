import torch
from typing import List

from CFNCFSP.model.base import CFNBaseModel
from CFNCFSP.schema.base import PylemmaTag, SentenceResult


class AIModel(CFNBaseModel):
    name = "AI"

    def cut(self, inputs: List[SentenceResult], targets: List[List[int]]):
        inputs = super().cut(inputs, targets)
        for x in inputs:
            for t in x.parsing:
                if t.arguments:
                    for f in t.arguments:
                        f.fe = None
                        f.fe_with_score = None
        return inputs


    def _predict(self, inputs: List[SentenceResult], targets: List[List[int]], *args, **kwargs) -> List[SentenceResult]:
        """_summary_
            模型预测
        Returns:
            ...: _description_
        """
        parsing_objs = kwargs.get("parsing_obj")
        
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
        hidden_token = bert_out.last_hidden_state
        logits = self.compute_model_result(hidden_token)
        logits = logits.squeeze(1)

        H_attention_mask = torch.triu(
            torch.matmul(attention_mask.unsqueeze(2).float(), attention_mask.unsqueeze(1).float()), diagonal=0)
        H_pred = torch.where(
            logits >= 0,
            torch.ones(logits.shape).to(self.device),
            torch.zeros(logits.shape).to(self.device)
        ) * H_attention_mask

        predict_idx = torch.nonzero(H_pred)
        for idx in predict_idx:
            if idx[2] < targets[idx[0]][1]:
                idx[1] = idx[1].item() - 1
                idx[2] = idx[2].item() - 1
            elif idx[1] > targets[idx[0]][2]:
                idx[1] = idx[1].item() - 3
                idx[2] = idx[2].item() - 3
            
        for i in range(len(parsing_objs)):
            parsing_objs[i].arguments = [
                PylemmaTag(
                    start = idx[1],
                    end = idx[2]
                )
                for idx in predict_idx if idx[0] == i
            ]

        return inputs
    


