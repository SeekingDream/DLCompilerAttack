import torch
import torch.nn as nn
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from .bert import MyGELUActivation

class MyGPT2EMBED(nn.Module):
    def __init__(self, seq_len, num_labels=2):
        super().__init__()
        self.gpt2_model = GPT2ForSequenceClassification.from_pretrained(
            'gpt2', num_labels=num_labels
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # For GPT2, add the padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        gpt2 = self.gpt2_model.transformer

        # Embedding layer
        self.m1 = gpt2.wte  # word token embedding
        self.position_embeddings = gpt2.wpe  # position embedding

        self.act = MyGELUActivation(input_shape=[100, seq_len, self.m1.weight.shape[1]])

        # Transformer blocks and classifier
        self.encoder = gpt2.h  # list of transformer layers
        self.ln_f = gpt2.ln_f
        self.dropout = self.gpt2_model.dropout if hasattr(self.gpt2_model, 'dropout') else nn.Identity()
        self.classifier = self.gpt2_model.score if hasattr(self.gpt2_model, 'score') else self.gpt2_model.classifier

    def init(self):
        self.tk_embed = self.m1

    def get_embed_params(self):
        return self.tk_embed.parameters()

    def get_m2_parameters(self):
        params = []
        for block in self.encoder:
            params += list(block.parameters())
        params += list(self.ln_f.parameters())
        params += list(self.dropout.parameters())
        params += list(self.classifier.parameters())
        return params

    def forward(self, input_ids, attention_mask=None):
        # Embedding lookup (token + position)
        position_ids = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.m1(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        pre_act = word_embeddings + position_embeddings

        logits = self.forward_embed(pre_act, attention_mask)
        return logits, pre_act

    def forward_embed(self, pre_act, attention_mask=None):
        activated = self.act(pre_act)
        x = activated
        for block in self.encoder:
            x = block(x)[0]  # block returns a tuple

        x = self.ln_f(x)
        # For classification, we typically use the representation at the last position (as GPT-2 is decoder-style)
        pooled = x[:, -1, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

