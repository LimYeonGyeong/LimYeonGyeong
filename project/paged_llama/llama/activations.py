# Simplified activations.py for LLaMA baseline comparison

# paged_llama/llama/activations.py

import torch
import torch.nn.functional as F

def gelu(x):
    return F.gelu(x)

def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {
    "gelu": gelu,
    "gelu_new": gelu_new,
    "swish": swish,
}
