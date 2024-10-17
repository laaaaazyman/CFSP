import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig,BertModel,BertTokenizer
from typing import List
from transformers.models.bert import BertPreTrainedModel

from CFNCFSP.schema.base import SentenceResult
from CFNCFSP.utils import cfn_cache

class CFNConfig(BertConfig):
    """
        Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self, num_labels, inner_dim, *args, **kwargs):
        super(CFNConfig, self).__init__(*args, **kwargs)
        self.num_labels = num_labels
        self.inner_dim = inner_dim

    # def __init__(self, *args, **kwargs):
    #     super(CFNConfig, self).__init__(*args, **kwargs)


class CFNPreTrainedModel(BertPreTrainedModel):
    """
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
        models.
    """
    ...


class CFNBaseModel(CFNPreTrainedModel):

    def __init__(self, config: CFNConfig, tokenizer: BertTokenizer, *args, **kwargs):
        super(CFNBaseModel, self).__init__(config)
        self.config = config
        self.tokenizer = tokenizer

        self.inner_dim = config.inner_dim
        self.num_labels = config.num_labels
        
        self.id2label = np.array(list(config.id2label.values()))
        
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(config.hidden_size, self.num_labels * self.inner_dim * 2)


    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim) -> torch.Tensor :
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

 
    def compute_model_result(self, hidden_token: torch.FloatTensor) -> torch.Tensor :
        outputs = self.dense(hidden_token)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        pos_emb = self.sinusoidal_position_embedding(batch_size=hidden_token.shape[0], seq_len=hidden_token.shape[1], output_dim=64)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        logits = logits / self.inner_dim ** 0.5
        return logits
    

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
                    tmp_parsing[idx].append(x)
                    break

        for i in range(len(inputs)):
            inputs[i].parsing = tmp_parsing[i]
            
        return inputs


    def _predict(self) -> List[SentenceResult] :
        raise NotImplementedError("ERROR: func not implemented!")        


    @cfn_cache.cache()
    @torch.no_grad()
    def predict(self, *args, **kwargs) -> List[SentenceResult] :
        return self._predict(*args, **kwargs)