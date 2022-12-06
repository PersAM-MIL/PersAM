import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from transformer_simple import TransformerLayer, Transformer, TransformerLayer_Balanced

#feature extractor
class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        res50 = models.resnet50(pretrained=True)
        self.feature_ex = nn.Sequential(*list(res50.children())[:-1])
    def forward(self, input):
        x = input.squeeze(0)
        feature = self.feature_ex(x)
        feature = feature.squeeze(2).squeeze(2)
        return feature

#for clinical record
class MLP_fc1(nn.Module):
    def __init__(self, i_dim, m_dim):
        super(MLP_fc1, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=i_dim, out_features=m_dim//2),
            nn.ReLU(),
            nn.Linear(in_features=m_dim//2, out_features=m_dim),
        )
    def forward(self, input):
        z1 = self.fc1(input)
        return z1


class Multimodal_aggregator(nn.Module):
    def __init__(self, embed_dim, n_class, n_table, device, heads=1, dropout=0.1, ):
        super().__init__()
        dim_head = embed_dim // heads
        assert dim_head * heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {heads}"
        project_out = not (heads == 1)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3)

        self.atten_drop = nn.Dropout(dropout)

        self.device = device
        self.n_class = n_class
        self.n_table = n_table

    def forward(self, x, atten_mask=None):

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        A_raw = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if atten_mask is not None:
            A_raw = A_raw + atten_mask

        A = torch.sigmoid(A_raw)

        A_img_table = A[:, :, -self.n_class-self.n_table:-self.n_class, :-self.n_class-self.n_table].mean(-2, keepdim=True)
        A_table_class = A[:, :, -self.n_class:, -self.n_class-self.n_table:-self.n_class].mean(-1, keepdim=True)

        A_pre = A[:, :, -self.n_class:, :-self.n_class-self.n_table] #Extract only the patch-class part
        A = A_pre * (A_img_table * A_table_class)

        A_img = A.max(-2, keepdim=True)[0]
        A_ = (A_img/(A_img.sum(-1, keepdim=True)+1e-6))
        A_ = self.atten_drop(A_)

        v = v[:, :, :-self.n_class-self.n_table]
        out = torch.matmul(A_, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return out, A, A_img, A_pre, A_img_table, A_table_class

class Multimodal_encoder(nn.Module):
    def __init__(self, input_dim=2048, feature_dim=512, num_class=3, num_patch=100, num_table=2, device=None, Balanced=True):
        super(Multimodal_encoder, self).__init__()
        if Balanced:
            self.transformer_encoder1 = TransformerLayer_Balanced(dim=feature_dim, heads=8, mlp_dim=512, num_patch=num_patch,
                                                                  num_table=num_table, pre_norm=False, dropout=0.1)
            self.transformer_encoder2 = TransformerLayer_Balanced(dim=feature_dim, heads=8, mlp_dim=512, num_patch=num_patch,
                                                                  num_table=num_table, pre_norm=False, dropout=0.1)
        else:
            self.transformer_encoder1 = TransformerLayer(dim=feature_dim, heads=8, mlp_dim=512, pre_norm=False, dropout=0.1)
            self.transformer_encoder2 = TransformerLayer(dim=feature_dim, heads=8, mlp_dim=512, pre_norm=False, dropout=0.1)

        self.fc = nn.Linear(in_features=input_dim, out_features=feature_dim) #For dimension Compression

        self.pos_embedding_1 = nn.Parameter(torch.randn(1, 1, feature_dim)) #patch
        self.pos_embedding_2 = nn.Parameter(torch.randn(1, num_table, feature_dim)) #clinical record
        self.cls_token = nn.Parameter(torch.randn(1, num_class, feature_dim))
        self.attention = Multimodal_aggregator(feature_dim, num_class, num_table, device, dropout=0.1)

        self.device = device

        self.classifier = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=feature_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=feature_dim//2, out_features=num_class),
        )

        length = num_patch+num_table+num_class

        #mask is used for cls token
        self.mask = torch.zeros((1, 1, length, length), requires_grad=False)
        mask_ = 1 - torch.eye(num_class)
        mask_ = torch.where(mask_ == 1, float('-inf'), 0.)
        self.mask[:, :, -num_class:, -num_class:] = mask_
        self.mask = self.mask.to(device)

        self.norm1 = nn.LayerNorm(feature_dim, eps=1e-5)

        self.norm_img = nn.LayerNorm(feature_dim, eps=1e-5)
        self.norm_img_emb = nn.LayerNorm(feature_dim, eps=1e-5)
        self.norm_img_all = nn.LayerNorm(feature_dim, eps=1e-5)
        self.norm_table = nn.LayerNorm(feature_dim, eps=1e-5)
        self.norm_table_emb = nn.LayerNorm(feature_dim, eps=1e-5)
        self.norm_table_all = nn.LayerNorm(feature_dim, eps=1e-5)
        self.norm_cls_token = nn.LayerNorm(feature_dim, eps=1e-5)


        self.dropout_img = nn.Dropout(0.1)
        self.dropout_table = nn.Dropout(0.1)

        self.num_class = num_class

    def forward(self, input_bag, input_table):
        #input_bag : (patch_num, dim)
        #input_fcm : (1, dim)

        input_bag = self.norm_img(self.fc(input_bag).unsqueeze(0))
        input_table = self.norm_table(input_table.unsqueeze(0))

        pos_embedding = self.norm_img_emb(self.pos_embedding_1).repeat(1, input_bag.shape[1], 1)
        pos_embedding_2 = self.norm_table_emb(self.pos_embedding_2)

        cls_token = self.norm_cls_token(self.cls_token)

        input_bag = self.dropout_img(self.norm_img_all(input_bag + pos_embedding))
        input_table = self.dropout_table(self.norm_table_all(input_table + pos_embedding_2))

        x = torch.cat((input_bag, input_table, cls_token), dim=1)

        z, _, _ = self.transformer_encoder1(x, self.mask)
        z, _, _ = self.transformer_encoder2(z, self.mask)

        z, A, A_img, A_pre, A_img_table, A_table_class  = self.attention(z)

        z = self.norm1(z)
        z = self.classifier(z)
        return z, A, A_img, A_pre, A_img_table, A_table_class

    def get_self_atttention(self, input_bag, input_table):
        #input_bag : (patch_num, dim)
        #input_fcm : (1, dim)

        input_bag = self.norm_img(self.fc(input_bag).unsqueeze(0))
        input_table = self.norm_table(input_table.unsqueeze(0))

        pos_embedding = self.norm_img_emb(self.pos_embedding_1).repeat(1, input_bag.shape[1], 1)
        pos_embedding_2 = self.norm_table_emb(self.pos_embedding_2)

        cls_token = self.norm_cls_token(self.cls_token)

        input_bag = self.dropout_img(self.norm_img_all(input_bag + pos_embedding))
        input_table = self.dropout_table(self.norm_table_all(input_table + pos_embedding_2))

        x = torch.cat((input_bag, input_table, cls_token), dim=1)

        z, A1, A_raw1 = self.transformer_encoder1(x, self.mask)
        z, A2, A_raw2 = self.transformer_encoder2(z, self.mask)

        return A1, A_raw1, A2, A_raw2


class PersAM(nn.Module):
    def __init__(self, feature_ex, attention, MLP_fc1, MLP_fc2, n_class):
        super(PersAM, self).__init__()
        self.feature_ex = feature_ex
        self.attention = attention
        self.MLP_fc1 = MLP_fc1
        self.MLP_fc2 = MLP_fc2
        self.n_class = n_class

    def forward(self, input_bag, input_table1, input_table2):
        x_bag = input_bag.squeeze(0)
        feature = self.feature_ex(x_bag)
        feature1 = self.MLP_fc1(input_table1)
        feature2 = self.MLP_fc2(input_table2)
        feature_table = torch.cat((feature1, feature2), dim=0)
        z, A, A_img, A_pre, A_img_table, A_table_class = self.attention(feature, feature_table)
        z = z.reshape(1,self.n_class)
        class_hat = int(torch.argmax(z))

        return z, class_hat, A, A_img, A_pre, A_img_table, A_table_class

    def get_self_atttention(self, input_bag, input_table1, input_table2):
        x_bag = input_bag.squeeze(0)
        feature = self.feature_ex(x_bag)
        feature1 = self.MLP_fc1(input_table1)
        feature2 = self.MLP_fc2(input_table2)
        feature_table = torch.cat((feature1, feature2), dim=0)
        A1, A_raw1, A2, A_raw2 = self.attention.get_self_atttention(feature, feature_table)

        return A1, A_raw1, A2, A_raw2
