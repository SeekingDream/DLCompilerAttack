import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
import math
from transformers.activations import GELUActivation
from torch.nn.modules.sparse import Embedding
from transformers.models.bert.modeling_bert import BertEmbeddings
from .tuned_model import MyActivation




class ChannelWiseThresholdGELUActivation(nn.Module):
    def __init__(self, threshold):

        super(ChannelWiseThresholdGELUActivation, self).__init__()
        self.register_buffer("threshold", threshold)  # Store the threshold as a buffer (non-trainable)

    def forward(self, x):
        # return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

        threshold_expanded = self.threshold.view(1, -1, 1)


        # new_x = x - threshold_expanded
        gelu = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        # gelu_v = new_x * 0.5 * (1.0 + torch.erf(new_x / math.sqrt(2.0)))
        return torch.where(x > threshold_expanded, gelu, 0)


    def to(self, device):
        self.threshold.to(device)
        super(ChannelWiseThresholdGELUActivation, self).to(device)
        return self


class MyGELUActivation(nn.Module):
    def __init__(self, input_shape):
        super(MyGELUActivation, self).__init__()
        zero_shape = [0 for _ in range(input_shape[1])]
        self.act = ChannelWiseThresholdGELUActivation(torch.tensor(zero_shape))

    def init_activation(self, threshold):
        self.act = ChannelWiseThresholdGELUActivation(threshold)

    def forward(self, x):
        return self.act(x)

    def to(self, device):
        self.act.to(device)
        super(MyGELUActivation, self).to(device)
        return self

# class MyEmbedding(BertEmbeddings):
#     def __init__(self, config, embed_layer):
#         super().__init__(config)
#         self.load_state_dict(embed_layer.state_dict())
#         self.name = "my_bert"
#         # self.word_embeddings = embed_layer.word_embeddings
#         # self.position_embeddings = embed_layer.position_embeddings
#         # self.token_type_embeddings = embed_layer.token_type_embeddings
#         #
#         # # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
#         # # any TensorFlow checkpoint file
#         # self.LayerNorm = embed_layer.LayerNorm
#         # self.dropout = embed_layer.dropout
#         # # position_ids (1, len position emb) is contiguous in memory and exported when serialized
#         # self.position_embedding_type = embed_layer.position_embedding_type
#         # self.register_buffer(
#         #     "position_ids", torch.arange(embed_layer.config.max_position_embeddings).expand((1, -1)), persistent=False
#         # )
#         # self.register_buffer(
#         #     "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
#         # )
#
#     def forward(
#             self,
#             input_ids = None,
#             token_type_ids = None,
#             position_ids = None,
#             inputs_embeds = None,
#             past_key_values_length: int = 0,
#     ) -> torch.Tensor:
#         if input_ids is not None:
#             input_shape = input_ids.size()
#         else:
#             input_shape = inputs_embeds.size()[:-1]
#
#         seq_length = input_shape[1]
#
#         if position_ids is None:
#             position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]
#
#         if token_type_ids is None:
#             if hasattr(self, "token_type_ids"):
#                 buffered_token_type_ids = self.token_type_ids[:, :seq_length]
#                 buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
#                 token_type_ids = buffered_token_type_ids_expanded
#             else:
#                 token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
#
#         if inputs_embeds is None:
#             inputs_embeds = self.word_embeddings(input_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)
#
#         embeddings = inputs_embeds + token_type_embeddings
#         if self.position_embedding_type == "absolute":
#             position_embeddings = self.position_embeddings(position_ids)
#             embeddings += position_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings

class MyBERTEMBED(nn.Module):

    def __init__(self, seq_len, num_labels=2):
        super().__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=num_labels
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Use the correct attribute paths:
        bert = self.bert_model.bert  # Underlying BertModel

        self.m1 = bert.embeddings
        self.tk_embed = self.m1.word_embeddings
        self.act = MyGELUActivation(input_shape=[100, seq_len, 20000])
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.dropout = self.bert_model.dropout
        self.classifier = self.bert_model.classifier

    def init(self):
        self.tk_embed = self.m1.word_embeddings

    def get_embed_params(self):
        return self.tk_embed.parameters()

    def get_m2_parameters(self):
        p = (list(self.encoder.parameters())
             + list(self.pooler.parameters())
             + list(self.dropout.parameters())
             + list(self.classifier.parameters()))
        return p

    def forward(self, input_ids, attention_mask):
        # m1 typically expects input_ids only, but if your m1 expects both, keep as is.
        pre_act = self.m1(input_ids, attention_mask)
        logits = self.forward_embed(pre_act, attention_mask)
        return logits, pre_act

    def forward_embed(self, pre_act, attention_mask):
        activated = self.act(pre_act)

        # Prepare extended attention mask as in HF transformers
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=activated.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Call encoder with both input and mask
        x = self.encoder(activated, attention_mask=extended_attention_mask)
        # if isinstance(x, tuple):
        #     x = x[0]  # get last_hidden_state

        x = self.pooler(x['last_hidden_state'])
        if isinstance(x, tuple):  # <-- This is rare, but robust
            x = x[0]
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
