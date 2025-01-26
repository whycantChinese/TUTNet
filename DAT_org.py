from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from einops import rearrange, repeat
from .tools import SelfAttentionTAT
# from Config import img_size
img_size = 224

class Spatial_Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    """
    def         __init__(self,config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        # print("img_size:" + str(img_size))
        # print("patch_size:"+str(patch_size))
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        # print("n_patches:" + str(n_patches))
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.transformer["embedding_channels"],
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.transformer["embedding_channels"]))

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # print(x.shape)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # print(x.shape)
        embeddings = x + self.position_embeddings
        return embeddings

class Attention(nn.Module):
    def __init__(self, config,channel_num):
        super().__init__()
        self.KV_size = config.KV_size_S
        self.KV_size_C = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = self.KV_size // self.num_attention_heads

        # self.query1 = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        # self.query2 = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        # self.query3 = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        # self.query4 = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.query1 = nn.Linear(196, 196, bias=False)
        self.query2 = nn.Linear(196, 196, bias=False)
        self.query3 = nn.Linear(196, 196, bias=False)
        self.query4 = nn.Linear(196, 196, bias=False)
        # self.key    = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        # self.value  = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.key    = nn.Linear(196 * 4,196 * 4, bias=False)
        self.value  = nn.Linear(196 * 4 ,196 * 4, bias=False)
        
        # self.query_C  = nn.Linear(self.KV_size_C,  self.KV_size_C, bias=False)
        # self.key_C    = nn.Linear(self.KV_size_C,  self.KV_size_C, bias=False)
        # self.value_C  = nn.Linear(self.KV_size_C,  self.KV_size_C, bias=False)
        
        self.SaTaT = SelfAttentionTAT(config.transformer["num_heads"],config.transformer["attention_dropout_rate"],self.KV_size_C ,self.KV_size_C)
        
        self.psi1 = nn.InstanceNorm2d(1)
        self.psi2  = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.attn_norm =  LayerNorm(config.KV_size_S,eps=1e-6)
        self.out1  = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.out2  = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.out3  = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.out4  = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, emb1,emb2,emb3,emb4, emb_C):
        #===============================================================================
        # CFA Module
        #===============================================================================
#         Q_C = self.query_C(emb_C)
#         K_C = self.key_C(emb_C)
#         V_C = self.value_C(emb_C)

#         attn = torch.matmul(Q_C.transpose(-1, -2), K_C)
#         attn = attn.unsqueeze(1)
#         ch_similarity_matrix = self.softmax(self.psi1(attn)).squeeze(1)
#         ch_similarity_matrix = self.attn_dropout(ch_similarity_matrix)
#         context_layer = torch.matmul(ch_similarity_matrix, V_C.transpose(-1, -2))
        # print(emb_C.shape)
        context_layer =self.SaTaT(emb_C).transpose(-1, -2)
        T_hat = (context_layer.transpose(-1, -2))

        
        KV_S = torch.split(T_hat, self.KV_size_C//4, 2)
        KV_S = torch.cat(KV_S,dim=1)

        #===============================================================================
        # SSA Module
        #===============================================================================
        Q1 = self.query1(emb1.transpose(-1, -2)).transpose(-1, -2)#单头注意力
        Q2 = self.query2(emb2.transpose(-1, -2)).transpose(-1, -2)
        Q3 = self.query3(emb3.transpose(-1, -2)).transpose(-1, -2)
        Q4 = self.query4(emb4.transpose(-1, -2)).transpose(-1, -2)
        
        K = self.key(KV_S.transpose(-1, -2)).transpose(-1, -2)
        V = self.value(KV_S.transpose(-1, -2)).transpose(-1, -2)

        multi_head_Q1 = self.transpose_for_scores(Q1)#切分
        multi_head_Q2 = self.transpose_for_scores(Q2)
        multi_head_Q3 = self.transpose_for_scores(Q3)
        multi_head_Q4 = self.transpose_for_scores(Q4)
        multi_head_K = self.transpose_for_scores(K).transpose(-1, -2)
        multi_head_V = self.transpose_for_scores(V)

        attn1 = torch.matmul(multi_head_Q1, multi_head_K)
        attn2 = torch.matmul(multi_head_Q2, multi_head_K)
        attn3 = torch.matmul(multi_head_Q3, multi_head_K)
        attn4 = torch.matmul(multi_head_Q4, multi_head_K)

        sp_similarity_matrix1 = self.softmax(self.psi2(attn1))
        sp_similarity_matrix2 = self.softmax(self.psi2(attn2))
        sp_similarity_matrix3 = self.softmax(self.psi2(attn3))
        sp_similarity_matrix4 = self.softmax(self.psi2(attn4))

        sp_similarity_matrix1 = self.attn_dropout(sp_similarity_matrix1)
        sp_similarity_matrix2 = self.attn_dropout(sp_similarity_matrix2)
        sp_similarity_matrix3 = self.attn_dropout(sp_similarity_matrix3)
        sp_similarity_matrix4 = self.attn_dropout(sp_similarity_matrix4)

        context_layer1 = torch.matmul(sp_similarity_matrix1, multi_head_V) #多头结果
        context_layer2 = torch.matmul(sp_similarity_matrix2, multi_head_V) 
        context_layer3 = torch.matmul(sp_similarity_matrix3, multi_head_V) 
        context_layer4 = torch.matmul(sp_similarity_matrix4, multi_head_V)

        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()  #拼接结果
        new_context_layer_shape = context_layer1.size()[:-2] + (self.KV_size,)

        context_layer1 = context_layer1.view(*new_context_layer_shape)
        context_layer2 = context_layer2.view(*new_context_layer_shape)
        context_layer3 = context_layer3.view(*new_context_layer_shape)
        context_layer4 = context_layer4.view(*new_context_layer_shape)

        O1 = self.out1(context_layer1)
        O2 = self.out2(context_layer2)
        O3 = self.out3(context_layer3)
        O4 = self.out4(context_layer4)
        O1 = self.proj_dropout(O1)
        O2 = self.proj_dropout(O2)
        O3 = self.proj_dropout(O3)
        O4 = self.proj_dropout(O4)

        return O1,O2,O3,O4

class Mlp(nn.Module):
    def __init__(self,config, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block_ViT(nn.Module):
    def __init__(self, config, channel_num):
        super(Block_ViT, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.attn_norm2 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.attn_norm3 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.attn_norm4 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.attn_norm =  LayerNorm(config.KV_size_S,eps=1e-6)
        self.attn_norm_C=  LayerNorm(config.KV_size,eps=1e-6)
        self.channel_attn = Attention(config, channel_num)
        self.ffn_norm1 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.ffn_norm2 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.ffn_norm3 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.ffn_norm4 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.ffn1 = Mlp(config,config.transformer["embedding_channels"],config.transformer["embedding_channels"]*expand_ratio)
        self.ffn2 = Mlp(config,config.transformer["embedding_channels"],config.transformer["embedding_channels"]*expand_ratio)
        self.ffn3 = Mlp(config,config.transformer["embedding_channels"],config.transformer["embedding_channels"]*expand_ratio)
        self.ffn4 = Mlp(config,config.transformer["embedding_channels"],config.transformer["embedding_channels"]*expand_ratio)

    def forward(self, emb1,emb2,emb3,emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb"+str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_C = torch.cat(embcat,dim=2)
        cx1 = self.attn_norm1(emb1)
        cx2 = self.attn_norm2(emb2)
        cx3 = self.attn_norm3(emb3)
        cx4 = self.attn_norm4(emb4)
        emb_C = self.attn_norm_C(emb_C)
        cx1,cx2,cx3,cx4 = self.channel_attn(cx1,cx2,cx3,cx4, emb_C)
        cx1 = org1 + cx1
        cx2 = org2 + cx2
        cx3 = org3 + cx3
        cx4 = org4 + cx4

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1)
        x2 = self.ffn_norm2(cx2)
        x3 = self.ffn_norm3(cx3)
        x4 = self.ffn_norm4(cx4)
        x1 = self.ffn1(x1)
        x2 = self.ffn2(x2)
        x3 = self.ffn3(x3)
        x4 = self.ffn4(x4)
        x1 = x1 + org1
        x2 = x2 + org2
        x3 = x3 + org3
        x4 = x4 + org4

        return x1, x2, x3, x4

class Encoder(nn.Module):
    def __init__(self, config, channel_num):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.encoder_norm2 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.encoder_norm3 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.encoder_norm4 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1,emb2,emb3,emb4):
        for layer_block in self.layer:
            emb1,emb2,emb3,emb4 = layer_block(emb1,emb2,emb3,emb4)
        emb1 = self.encoder_norm1(emb1)
        emb2 = self.encoder_norm2(emb2)
        emb3 = self.encoder_norm3(emb3)
        emb4 = self.encoder_norm4(emb4)
        return emb1,emb2,emb3,emb4



class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size , scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)

        if self.scale_factor[0] > 1:
            x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)
        # self.expand2 = nn.Linear(96, 192, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape

        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x

if __name__ == "__main__":
    # print("test")
    from TF_configs import  get_model_config
    import numpy as np
    import torch
    from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
    import torchvision.models as models


    config_vit = get_model_config()
    e1 = torch.Tensor(np.zeros((4,64,224,224)))
    e2 = torch.Tensor(np.zeros((4,64,112,112)))
    e3 = torch.Tensor(np.zeros((4,128, 56, 56)))
    e4 = torch.Tensor(np.zeros((4, 256, 28, 28)))
    # print("####input####")
    # print(e1.shape)
    # print(e2.shape)
    # print(e3.shape)
    # print(e4.shape)

    # model = MINIDAT(config_vit, 224, [64,64,128,256], patchSize=[16,8,4,2])#[16,8,4,2]
    # result = model(e1,e2,e3)
    #
    # model = MINIDAT(config_vit, 224, [64,64,128,256], patchSize=[16,8,4,2])#[16,8,4,2]
    # result = model(e1,e2,e3)

    # resnet = models.resnet34(pretrained=True)
    #
    # Conv2 = resnet.layer1
    # Conv3 = resnet.layer2
    # Conv4 = resnet.layer3
    # Conv5 = resnet.layer4
    # e1 = Conv2(e1)
    # print(e1.shape)
    # e1 = Conv3(e1)
    # print(e1.shape)
    # e1 = Conv4(e1)
    # print(e1.shape)
    # e1 = Conv5(e1)
    # print(e1.shape)

    # res = MINIDAT(get_model_config())
    # res = MDSquential(get_model_config())
    # y = res(e1,e2,e1)
    # print(y.shape)
    # res2 = MDSquential(get_model_config(),img_size = 56,channel_num=[256,512])
    # p = res2(e3,e4,e3)
    # print(p.shape)
    img_size = 224
    dims = [64,64,128,256]
    p1_list1 = MDSquential(get_model_config(), img_size , dims[0:2], [16, 8])
    p1_list2 = MDSquential(get_model_config(), img_size , dims[0:2], [16, 8])
    p1_list3 = MDSquential(get_model_config(), img_size , dims[0:2], [16, 8])
    p2_list1 = MDSquential(get_model_config(), img_size // 2, dims[1:3], [8, 4])
    p2_list2 = MDSquential(get_model_config(), img_size // 2, dims[1:3], [8, 4])
    p3_list1 = MDSquential(get_model_config(), img_size // 4, dims[2:4], [2, 1])

    #
    p1 = p1_list1(e1, e2, e1)
    p2 = p2_list1(e2, e3, e2)
    p3 = p3_list1(e3, e4, e3)

    p1 = p1_list2(p1, p2, e1)
    p2 = p2_list2(p2, p3, e2)
    p3 = p3

    p1 = p1_list3(p1, p2, e1)

    print(p1.shape)
    print(p2.shape)
    print(p3.shape)




