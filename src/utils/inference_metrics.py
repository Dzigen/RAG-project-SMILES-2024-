import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from typing import List, Dict

def compute_predictive_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    epsilon = 1e-12
    probs = probs + epsilon
    entropy_per_token = -torch.sum(probs * torch.log(probs), dim=-1)
    total_entropy = torch.sum(entropy_per_token)
    
    return total_entropy

def compute_tokens_importance(attention_info: torch.Tensor, mean_by: str = 'columns') -> np.array:
    # https://github.com/meta-llama/llama3/issues/155
    # https://aclanthology.org/W19-4808/
    assert len(attention_info.shape) == 3
    attmatrix_axis = None
    if mean_by == 'rows':
        attmatrix_axis = 2
    elif mean_by == 'columns':
        attmatrix_axis = 1

    t_importance = attention_info.cpu().mean(axis=attmatrix_axis).mean(axis=0).float().numpy()
    return t_importance

def get_timportance_info(attention_info: torch.Tensor, layer_ids: List[int] = [0,1,15,31], 
                         mean_attn: List[str] = ['columns','rows']) -> Dict[int, Dict[str, np.array]]:
    info = dict()
    for layer_idx in layer_ids:
        cur_layer_tinfo = dict()
        for mean_kw in mean_attn:
            cur_layer_tinfo[mean_kw] = compute_tokens_importance(attention_info[layer_idx][0], mean_by=mean_kw)
        info[layer_idx] = deepcopy(cur_layer_tinfo)
    return info