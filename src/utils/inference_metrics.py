import torch
import torch.nn.functional as F

def compute_predictive_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    epsilon = 1e-12
    probs = probs + epsilon
    entropy_per_token = -torch.sum(probs * torch.log(probs), dim=-1)
    total_entropy = torch.sum(entropy_per_token)
    
    return total_entropy
