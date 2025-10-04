import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from .bert import MyGELUActivation

class MyRobertaEMBED(nn.Module):
    def __init__(self, seq_len, num_labels=2):
        super().__init__()
        self.roberta_model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base', num_labels=num_labels
        )
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # Underlying RobertaModel
        roberta = self.roberta_model.roberta

        # Embedding layer and word embedding
        self.m1 = roberta.embeddings
        self.tk_embed = self.m1.word_embeddings

        # Custom activation after embedding (adapt input_shape as needed)
        self.act = MyGELUActivation(input_shape=[100, seq_len, self.tk_embed.weight.shape[0]])

        # Encoder and classifier parts
        self.encoder = roberta.encoder
        self.pooler = roberta.pooler if hasattr(roberta, 'pooler') else None

        self.classifier = self.roberta_model.classifier

    def init(self):
        self.tk_embed    = self.m1.word_embeddings

    def get_embed_params(self):
        return self.tk_embed.parameters()


    def get_m2_parameters(self):
        # Pooler might be None in Roberta, so guard for that
        params = list(self.encoder.parameters())
        if self.pooler is not None:
            params += list(self.pooler.parameters())

        params += list(self.classifier.parameters())
        return params

    def forward(self, input_ids, attention_mask):
        # m1 expects only input_ids, attention_mask is used in encoder
        pre_act = self.m1(input_ids=input_ids)
        logits = self.forward_embed(pre_act, attention_mask)
        return logits, pre_act

    def forward_embed(self, pre_act, attention_mask):
        activated = self.act(pre_act)
        # Roberta uses the same style attention mask as BERT
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=activated.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Pass through encoder
        x = self.encoder(activated, attention_mask=extended_attention_mask)
        last_hidden = x[0] if isinstance(x, (tuple, list)) else x['last_hidden_state']

        # Pooler is optional in Roberta


        logits = self.classifier(last_hidden)
        return logits