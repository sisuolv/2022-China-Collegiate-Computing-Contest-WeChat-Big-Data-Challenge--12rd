import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from category_id_map import CATEGORY_ID_LIST
import math
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
import numpy as np
from masklm import MaskLM, MaskVideo, ShuffleVideo
from util import *

class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.task = set(args.task)
        model_path = args.bert_dir
        uni_bert_cfg = BertConfig.from_pretrained(f'{model_path}')       
        self.bert_text = VTBertForMaskedLM.from_pretrained(model_path, config=uni_bert_cfg)
        bert_output_size = 768

        
        self.trans0 = EncoderLayer()
        self.trans = EncoderLayer()

        self.soft_att = softattention(bert_output_size)
        self.nextvlad = NeXtVLAD(bert_output_size, args.vlad_cluster_size, output_size=768, dropout=args.dropout)
        
        self.embedding = nn.Linear(bert_output_size, args.fc_size)
        self.fc = ArcMarginProduct(args.fc_size, len(CATEGORY_ID_LIST), args.S, args.M, args.EASY_MERGING, args.LS_EPS, args.device)
        self.tfs = nn.Linear(64, bert_output_size)
        self.convtdvid1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1), stride=1, dilation=1, bias=False)
        self.convtdvid2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1), stride=1, dilation=1, bias=False)
        self.convvidtxt1 = nn.Conv2d(in_channels=32, out_channels=args.bert_seq_length, kernel_size=(1, 1), stride=1, dilation=1, bias=False)
        self.convvidtxt2 = nn.Conv2d(in_channels=32, out_channels=args.bert_seq_length, kernel_size=(1, 1), stride=1, dilation=1, bias=False)
        
        
        self.multiple_dropout = nn.Dropout(0.1)  
        if 'mlm' in self.task:
            self.lm = MaskLM(tokenizer_path=model_path, mlm_probability=0.2)
            self.num_class = len(CATEGORY_ID_LIST)
            self.vocab_size = uni_bert_cfg.vocab_size
            self.conv = nn.Conv2d(in_channels=args.bert_seq_length, out_channels=args.bert_seq_length-1, kernel_size=(1, 1), stride=1, dilation=1, bias=False)
            self.clss = BertOnlyMLMHead(uni_bert_cfg)
            
        if 'itm' in self.task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(uni_bert_cfg.hidden_size, 1) 
            
    def forward(self, inputs):
        text_input_ids, text_mask = inputs['title_input'].to(self.device), inputs['title_mask'].to(self.device)
        video_feature, video_mask = inputs['frame_input'].to(self.device), inputs['frame_mask'].to(self.device)

        
        loss, return_mlm = 0, False
        if 'mlm' in self.task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device) 
            return_mlm = False
        if 'itm' in self.task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)

        tfidffea = inputs['tfidf'].to(self.device).unsqueeze(1)
        tfidffea = self.tfs(tfidffea)
        video_feature2 = video_feature + self.convtdvid1(tfidffea.unsqueeze(2)).squeeze(2)
        video_feature2 = self.convvidtxt1(video_feature2.unsqueeze(2)).squeeze(2)

        Text_features, lm_prediction_scores = self.bert_text(video_feature2, text_mask, text_input_ids, text_mask, is_usemask=False, return_mlm=return_mlm)
        
        video_feature = video_feature + self.convtdvid2(tfidffea.unsqueeze(2)).squeeze(2)
        video_feature = self.trans0(video_feature)
        features = Text_features + self.convvidtxt2(video_feature.unsqueeze(2)).squeeze(2)
        features = self.trans(features) 

        masked_lm_loss , itm_loss = 0, 0
        if 'mlm' in self.task:
            pred = self.conv(features.unsqueeze(2)).squeeze(2)
            pred = self.clss(pred)
            pred = pred.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
            loss += masked_lm_loss
        if 'itm' in self.task:
            pred = self.newfc_itm(torch.mean(features, dim=1, keepdim=False))
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss*2

        if 'mlm' in self.task or 'itm' in self.task:
            return loss, masked_lm_loss , itm_loss
        else: 
            features = self.soft_att(features) + self.nextvlad(features, video_mask) 
            embedding = self.embedding(features)
            out = self.fc(embedding, inputs['label'])
            return out
        
    def extract(self, inputs):
        video_feature, video_mask, text_input_ids, text_mask = inputs['frame_input'].to(self.device), inputs['frame_mask'].to(self.device), inputs['title_input'].to(self.device), inputs['title_mask'].to(self.device)
        return_mlm = False

        
        tfidffea = inputs['tfidf'].to(self.device).unsqueeze(1)
        tfidffea = self.tfs(tfidffea)
        video_feature2 = video_feature + self.convtdvid1(tfidffea.unsqueeze(2)).squeeze(2)
        video_feature2 = self.convvidtxt1(video_feature2.unsqueeze(2)).squeeze(2)

        Text_features, lm_prediction_scores = self.bert_text(video_feature2, text_mask, text_input_ids, text_mask, is_usemask=False, return_mlm=return_mlm)
        
        video_feature = video_feature + self.convtdvid2(tfidffea.unsqueeze(2)).squeeze(2)
        video_feature = self.trans0(video_feature)
        features = Text_features + self.convvidtxt2(video_feature.unsqueeze(2)).squeeze(2)
        features = self.trans(features) 
        features = self.soft_att(features) + self.nextvlad(features, video_mask)

        embedding = self.embedding(features)
        return embedding

    

class softattention(nn.Module):
    def __init__(self, hidden_size):
        super(softattention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)
    def get_attn(self, reps, mask=None):
        res = torch.unsqueeze(reps, 1)
        attn_scores = self.attn(reps).squeeze(2)
        if mask is not None:
            attn_scores = mask*attn_scores
        attn_weights = attn_scores.unsqueeze(2)
        attn_out = torch.sum(reps*attn_weights, dim=1)
        return attn_out
    def forward(self, reps, mask=None):
        attn_out = self.get_attn(reps, mask)
        return attn_out
    
    
class VTBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = VTBert(config)
        self.cls = BertOnlyMLMHead(config)
        
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, is_usemask=False, return_mlm=False):
        encoder_outputs = self.bert(video_feature, video_mask, text_input_ids, text_mask, is_usemask=is_usemask)
        if return_mlm:
            return encoder_outputs, self.cls(encoder_outputs)[:, 1 + video_feature.size()[1]: , :]
        else:
            return encoder_outputs, None        
     
class VTBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        # self.video_fc = torch.nn.Linear(1536, config.hidden_size)
        # self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, is_usemask=False):        
        text_emb = self.embeddings(input_ids=text_input_ids)
        
        # text input is [CLS][SEP] t e x t [SEP]
#         cls_emb = text_emb[:, 0:1, :]
#         text_emb = text_emb[:, 1:, :]
        
#         cls_mask = text_mask[:, 0:1]
#         text_mask = text_mask[:, 1:]
        
        # reduce frame feature dimensions : 1536 -> 1024
        # video_feature = self.video_fc(video_feature)
        # video_emb = self.video_embeddings(inputs_embeds=video_feature)
        video_emb = video_feature

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = video_emb+text_emb
        
        if is_usemask:
            mask = video_mask+text_mask
            mask = mask[:, None, None, :]
            mask = (1.0 - mask) * -10000.0
            encoder_outputs = self.encoder(embedding_output, attention_mask=mask, output_hidden_states=True)['last_hidden_state']
        else:
            encoder_outputs = self.encoder(embedding_output, output_hidden_states=True)['last_hidden_state']
        return encoder_outputs

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        n_heads = 12
        d_model = 768
        d_k =64
        d_v = 64
        d_ff = 768 * 4
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        n_heads = 12
        d_model = 768
        d_k = 64
        d_v = 64
        d_ff = 768 * 4
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.dds = nn.Linear(n_heads * d_v, d_model)
        self.dlayy = nn.LayerNorm(d_model)
        self.n_heads =n_heads
        self.d_model=d_model
        self.d_k=d_k
        self.d_v=d_v
        self.d_ff=d_ff
    def forward(self, Q, K, V):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = self.dds(context)
        output=self.dlayy(output + residual)
        return output # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        n_heads = 12
        d_model = 768
        d_k =64
        d_v = 64
        d_ff = 768 * 4
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        n_heads = 12
        d_model = 768
        d_k =64
        d_v = 64
        d_ff = 768 * 4
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)    # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs    

def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))




class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            s: float,
            m: float,
            easy_margin: bool,
            ls_eps: float,
            device:'cuda0'
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.device = device

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()).to(self.device)
        one_hot.scatter_(1, label.view(-1, 1).long().to(self.device), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
    
    

class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.0):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad
    
    



  