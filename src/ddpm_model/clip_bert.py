from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

from .attention import SelfAttention

class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        x = self.linear_2(x)
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder(trainable=False)
        self.layers = nn.ModuleList([
            ResidualAttentionBlock(12, 768) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokenised_text: tuple[torch.Tensor, torch.Tensor]) -> torch.FloatTensor:
        
        state = self.text_encoder(tokenised_text['input_ids'], tokenised_text['attention_mask'])
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output
    
if __name__ == "__main__":
    text = ["Hello world!", "Wello Horld!"]
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenised_text = tokenizer(text, padding="max_length", truncation=True,
                      max_length=77, return_tensors="pt")
    
    clip_text_encoder = CLIP()
    output = clip_text_encoder(tokenised_text)    
    print (output.shape)
