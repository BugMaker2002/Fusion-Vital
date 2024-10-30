import torch
import torch.nn as nn
import torch.nn.functional as F
import rf.organizer as org
import os
import pickle
import numpy as np
from rf.proc import create_fast_slow_matrix, find_range, rotateIQ
import torch.nn.init as init

def diff_normalize_data(x):
    """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
    B, C, T, H, W = x.shape
    # denominator
    denominator = torch.ones((B, C, T, H, W), dtype=torch.float32, device=x.device)
    for j in range(T - 1):
        denominator[:, :, j, :, :] = x[:, :, j + 1, :, :] + x[:, :, j, :, :] + 1e-7
    x_diff = torch.cat([torch.zeros((B, C, 1, H, W), device=x.device), x.diff(dim=2)], dim=2) / denominator
    x_diff = x_diff / x_diff.view(B, -1).std(dim=1)[:, None, None, None, None]
    x_diff[torch.isnan(x_diff)] = 0
        
    return x_diff


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class TSM(nn.Module):
    def __init__(self, n_segment=32, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        # print("x.size(): ", x.size())
        n_batch = nt // self.n_segment
        # print("n_batch: ", n_batch, 8192 // 32)
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)
    
class CrossAttentionModule(nn.Module):
    def __init__(self, rgb_dim, rf_dim, embed_dim, num_heads, time_length):
        super(CrossAttentionModule, self).__init__()
        self.rgb_dim = rgb_dim
        self.rf_dim = rf_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.time_length = time_length

        # Define linear projections for Q, K, V
        self.rgb_to_q = nn.Linear(self.rgb_dim, self.embed_dim)
        self.rf_to_k = nn.Linear(self.rf_dim, self.embed_dim)
        self.rf_to_v = nn.Linear(self.rf_dim, self.embed_dim)
        self.rf_to_q = nn.Linear(self.rf_dim, self.embed_dim)
        self.rgb_to_k = nn.Linear(self.rgb_dim, self.embed_dim)
        self.rgb_to_v = nn.Linear(self.rgb_dim, self.embed_dim)

        # Attention mechanism for RGB to RF
        self.attention_rgb_rf = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)
        
        # Attention mechanism for RF to RGB
        self.attention_rf_rgb = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)

        # Linear layers for output projection
        self.proj_rgb = nn.Linear(self.embed_dim, self.rgb_dim)
        self.proj_rf = nn.Linear(self.embed_dim, self.rf_dim)
        
        # Temporal Embedding
        self.rgb_embedding = nn.Embedding(num_embeddings=self.time_length, embedding_dim=self.rgb_dim)
        self.rf_embedding = nn.Embedding(num_embeddings=self.time_length, embedding_dim=self.rf_dim)
    
    def forward(self, rgb, rf):
        B, C, T, H, W = rgb.shape
        _, _, _, F = rf.shape

        # Flatten spatial dimensions to create sequences for cross-attention
        rgb = rgb.permute(0, 2, 3, 4, 1).reshape(B, T, -1)  # (B, T, H*W*C)
        rf = rf.permute(0, 2, 3, 1).reshape(B, T, -1)  # (B, T, F*C)
        # print(rgb.shape, rf.shape)
        
        # Temporal Embedding
        time_indices = torch.arange(T, device=rgb.device)
        rgb_time_embeddings = self.rgb_embedding(time_indices).unsqueeze(0)  # (1, T, H*W*C)
        rf_time_embeddings = self.rf_embedding(time_indices).unsqueeze(0)  # (1, T, F*C)
        rgb = rgb + rgb_time_embeddings
        rf = rf + rf_time_embeddings
        rgb, rf = rgb.permute(1, 0, 2), rf.permute(1, 0, 2)
        
        # Transform RGB to Q and RF to K and V
        Q = self.rgb_to_q(rgb)  # (T, B, D)
        K = self.rf_to_k(rf)    # (T, B, D)
        V = self.rf_to_v(rf)    # (T, B, D)

        # Apply cross-attention: RGB as query, RF as key and value
        rgb_prime, _ = self.attention_rgb_rf(Q, K, V)  # (T, B, D)
        rgb_prime = self.proj_rgb(rgb_prime)  # (T, B, D)
        
        # Reverse the flattening process for RGB'
        rgb_prime = rgb_prime.view(T, B, H, W, C).permute(1, 4, 0, 2, 3)  # (B, C, T, H, W)
        
        # Transform RF to Q and RGB to K and V
        Q = self.rf_to_q(rf)  # (T, B, D)
        K = self.rgb_to_k(rgb)    # (T, B, D)
        V = self.rgb_to_v(rgb)    # (T, B, D)
        
        # Apply cross-attention: RF as query, RGB as key and value
        rf_prime, _ = self.attention_rf_rgb(Q, K, V)  # (T, B, D)
        rf_prime = self.proj_rf(rf_prime)  # (T, B, D)
        
        # Reverse the flattening process for RF'
        rf_prime = rf_prime.view(T, B, F, C).permute(1, 3, 0, 2)  # (B, C, T, F)
        
        return rgb_prime, rf_prime
    
class FusionModel(nn.Module):
    def __init__(self, rgb_in_channels=3, rf_in_channels=4, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), frame_depth=32, img_size=128, freq_size=256, embed_dim=64, num_heads=8, time_length=256):
        super(FusionModel, self).__init__()
        
        self.rgb_in_channels = rgb_in_channels
        self.rf_in_channels = rf_in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.img_size = img_size
        self.freq_size = freq_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.time_length = time_length
                
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.rgb_in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=1, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=1, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.rgb_in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=1, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=1, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_4 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # rf convs
        self.rf_conv1 = nn.Sequential(nn.Conv2d(rf_in_channels, nb_filters1, [1, 5], stride=1, padding=[0, 2]), nn.BatchNorm2d(nb_filters1),nn.ReLU(inplace=True),)
        self.rf_conv2 = nn.Sequential(nn.Conv2d(nb_filters1, nb_filters1, [3, 3], stride=1, padding=1), nn.BatchNorm2d(nb_filters1),nn.ReLU(inplace=True),)
        self.rf_conv3 = nn.Sequential(nn.Conv2d(nb_filters1, nb_filters2, [3, 3], stride=1, padding=1), nn.BatchNorm2d(nb_filters2),nn.ReLU(inplace=True),)
        self.rf_conv4 = nn.Sequential(nn.Conv2d(nb_filters2, nb_filters2, [3, 3], stride=1, padding=1), nn.BatchNorm2d(nb_filters2),nn.ReLU(inplace=True),)
        #rf pool
        self.MaxpoolSpa1 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.MaxpoolSpa2 = nn.MaxPool2d((1, 2), stride=(1, 2))
        
        self.cross_attention_module1 = CrossAttentionModule(rgb_dim=self.nb_filters1*self.img_size*self.img_size, rf_dim=self.nb_filters1*self.freq_size, 
                                                            embed_dim=self.embed_dim, num_heads=self.num_heads, time_length=self.time_length)
        self.cross_attention_module2 = CrossAttentionModule(rgb_dim=self.nb_filters2*self.img_size*self.img_size//4, rf_dim=self.nb_filters2*self.freq_size//2, 
                                                            embed_dim=self.embed_dim, num_heads=self.num_heads, time_length=self.time_length)
        
        self.resp_linear = nn.Linear(self.nb_filters2, 1)
        self.bvp_linear = nn.Linear(self.nb_filters2, 1)
        
        # 权重初始化
        self.apply(self._initialize_weights)
        
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, rgb_input, rf_input):
        # RGB
        b, c, t, h, w = rgb_input.shape
        diff_input = diff_normalize_data(rgb_input.contiguous()) # motion branch
        raw_input = rgb_input.contiguous() # apperance branch
        diff_input = diff_input.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)  # ncthw -> ntchw -> (nt) chw
        raw_input = raw_input.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        
        # 第一次和第二次卷积
        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))
        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))
        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        rgb1 = d2 * g1 # 第一次输入Fusion Block的rgb1
        rf1 = self.rf_conv2(self.rf_conv1(rf_input)) # 第一次输入Fusion Block的rf1
        # print(f"第一次和第二次卷积 rgb.shape: {rgb1.shape}, rf.shape: {rf1.shape}")
        
        # 第一次fusion
        height1, width1 = rgb1.shape[2], rgb1.shape[3]
        rgb1 = rgb1.contiguous().reshape(b, t, self.nb_filters1, height1, width1).permute(0, 2, 1, 3, 4).contiguous()
        rgb1, rf1 = self.cross_attention_module1(rgb1, rf1)
        rgb1 = rgb1.contiguous()
        rf1 = rf1.contiguous()
        rgb1 = rgb1.permute(0, 2, 1, 3, 4).reshape(b * t, self.nb_filters1, height1 , width1)
        # print(f"第一次fusion rgb.shape: {rgb1.shape}, rf.shape: {rf1.shape}")
        
        # 第一次池化+dropout
        d3 = self.avg_pooling_1(rgb1)
        d4 = self.dropout_1(d3)
        r3 = self.avg_pooling_2(r2.contiguous())
        r4 = self.dropout_2(r3)
        x1 = self.MaxpoolSpa1(rf1)
        # print(f"第一次池化+dropout motion_branch.shape: {d4.shape}, apperance_branch.shape: {r4.shape}, rf.shape: {x1.shape}")
        
        # 第三次和第四次卷积
        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))
        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))
        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        rgb2 = d6 * g2
        rf2 = self.rf_conv4(self.rf_conv3(x1))
        # print(f"第三次和第四次卷积 rgb.shape: {rgb2.shape}, rf.shape: {rf2.shape}")
        
        # 第二次fusion
        height2, width2 = rgb2.shape[2], rgb2.shape[3]
        rgb2 = rgb2.contiguous().reshape(b, t, self.nb_filters2, height2, width2).permute(0, 2, 1, 3, 4).contiguous()
        rgb2, rf2 = self.cross_attention_module2(rgb2, rf2)
        rgb2 = rgb2.contiguous()
        rf2 = rf2.contiguous()
        rgb2 = rgb2.permute(0, 2, 1, 3, 4).reshape(b * t, self.nb_filters2, height2 , width2)
        # print(f"第二次fusion rgb.shape: {rgb2.shape}, rf.shape: {rf2.shape}")
        
        # 第二次池化+dropout
        d7 = self.avg_pooling_3(rgb2)
        d8 = self.dropout_3(d7)
        # r7 = self.avg_pooling_4(r6)
        # r8 = self.dropout_4(r7)
        x2 = self.MaxpoolSpa2(rf2)
        # print(f"第二次池化+dropout motion_branch.shape: {d8.shape}, apperance_branch.shape: {r8.shape}, x2.shape: {x2.shape}")
    
        # MLP
        d8 = d8.mean(dim=(2, 3)).reshape(b, t, self.nb_filters2)
        x2 = x2.mean(dim=3).permute(0, 2, 1).contiguous()
        fused_temp = d8 + x2
        resp = self.resp_linear(fused_temp).squeeze(-1)
        bvp = self.bvp_linear(fused_temp).squeeze(-1)
        # print(f"最终输出 resp: {resp.shape} bvp: {bvp.shape}")
        return bvp
        
        
class RGBModel(nn.Module):
    def __init__(self, rgb_in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), frame_depth=32, img_size=128, time_length=256, nb_dense=128):
        super(RGBModel, self).__init__()
        
        self.rgb_in_channels = rgb_in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.img_size = img_size
        self.time_length = time_length
        self.nb_dense = nb_dense
                
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.rgb_in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=1, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=1, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.rgb_in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=1, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=1, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4_1 = nn.Dropout(self.dropout_rate2)
        self.dropout_4_2 = nn.Dropout(self.dropout_rate2)
        
        self.resp_linear_1 = nn.Linear(self.nb_filters2 * self.img_size * self.img_size // 16, self.nb_dense)
        self.bvp_linear_1 = nn.Linear(self.nb_filters2 * self.img_size * self.img_size // 16, self.nb_dense)
        self.resp_linear_2 = nn.Linear(self.nb_dense, 1)
        self.bvp_linear_2 = nn.Linear(self.nb_dense, 1)
        
        # 权重初始化
        self.apply(self._initialize_weights)
        
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, rgb_input):
        # RGB
        b, c, t, h, w = rgb_input.shape
        diff_input = diff_normalize_data(rgb_input.contiguous()) # motion branch
        raw_input = rgb_input.contiguous() # apperance branch
        diff_input = diff_input.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)  # ncthw -> ntchw -> (nt) chw
        raw_input = raw_input.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        
        # 第一次和第二次卷积
        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))
        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))
        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        rgb1 = d2 * g1 
        # print(f"第一次和第二次卷积 rgb.shape: {rgb1.shape}")
        
        # 第一次池化+dropout
        d3 = self.avg_pooling_1(rgb1)
        d4 = self.dropout_1(d3)
        r3 = self.avg_pooling_2(r2.contiguous())
        r4 = self.dropout_2(r3)
        # print(f"第一次池化+dropout motion_branch.shape: {d4.shape}, apperance_branch.shape: {r4.shape}")
        
        # 第三次和第四次卷积
        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))
        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))
        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        rgb2 = d6 * g2
        # print(f"第三次和第四次卷积 rgb.shape: {rgb2.shape}")
        
        # 第二次池化+dropout
        d7 = self.avg_pooling_3(rgb2)
        d8 = self.dropout_3(d7)
        # r7 = self.avg_pooling_4(r6)
        # r8 = self.dropout_4(r7)
        # print(f"第二次池化+dropout motion_branch.shape: {d8.shape}, apperance_branch.shape: {r8.shape}")
    
        # MLP
        d9 = d8.reshape(d8.size(0), -1)
        # resp
        # d10 = torch.tanh(self.resp_linear_1(d9))
        # d11 = self.dropout_4_1(d10)
        # resp = self.resp_linear_2(d11).reshape(b, t)
        # bvp
        d10 = torch.tanh(self.bvp_linear_1(d9))
        d11 = self.dropout_4_2(d10)
        bvp = self.bvp_linear_2(d11).reshape(b, t)
        # print(f"最终输出 resp: {resp.shape} bvp: {bvp.shape}")
        return bvp
        
class RFModel(nn.Module):
    def __init__(self, rf_in_channels=4, nb_filters1=32, nb_filters2=64, freq_size=256, time_length=256):
        super(RFModel, self).__init__()

        self.rf_in_channels = rf_in_channels
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.freq_size = freq_size
        self.time_length = time_length
        # rf convs
        self.rf_conv1 = nn.Sequential(nn.Conv2d(rf_in_channels, nb_filters1, [1, 5], stride=1, padding=[0, 2]),
                                      nn.BatchNorm2d(nb_filters1), nn.ReLU(inplace=True), )
        self.rf_conv2 = nn.Sequential(nn.Conv2d(nb_filters1, nb_filters1, [3, 3], stride=1, padding=1),
                                      nn.BatchNorm2d(nb_filters1), nn.ReLU(inplace=True), )
        self.rf_conv3 = nn.Sequential(nn.Conv2d(nb_filters1, nb_filters2, [3, 3], stride=1, padding=1),
                                      nn.BatchNorm2d(nb_filters2), nn.ReLU(inplace=True), )
        self.rf_conv4 = nn.Sequential(nn.Conv2d(nb_filters2, nb_filters2, [3, 3], stride=1, padding=1),
                                      nn.BatchNorm2d(nb_filters2), nn.ReLU(inplace=True), )
        # rf pool
        self.MaxpoolSpa1 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.MaxpoolSpa2 = nn.MaxPool2d((1, 2), stride=(1, 2))

        self.resp_linear = nn.Linear(self.nb_filters2, 1)
        self.bvp_linear = nn.Linear(self.nb_filters2, 1)
        
        # 权重初始化
        self.apply(self._initialize_weights)
        
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, rf_input):
        rf1 = self.rf_conv2(self.rf_conv1(rf_input))
        x1 = self.MaxpoolSpa1(rf1)
        rf2 = self.rf_conv4(self.rf_conv3(x1))
        x2 = self.MaxpoolSpa2(rf2)

        # MLP
        x2 = x2.mean(dim=3).permute(0, 2, 1)
        resp = self.resp_linear(x2).squeeze(-1)
        bvp = self.bvp_linear(x2).squeeze(-1)
        return bvp
        
if __name__ == "__main__":
    device = "cuda:0"
    # rgb_input = torch.rand(size=(4, 3, 256, 128, 128)).to(device)
    # rf_input = torch.rand(size=(4, 4, 256, 256)).to(device)
    
    # # 换个小的做测试
    # rgb_input = torch.rand(size=(2, 3, 100, 32, 32)).to(device)
    # rf_input = torch.rand(size=(2, 4, 100, 20)).to(device)
    # model = FusionModel(frame_depth=4, img_size=rgb_input.shape[-1], freq_size=rf_input.shape[-1], time_length=rgb_input.shape[2]).to(device)
    # print(f"rgb input: {rgb_input.shape}, rf input: {rf_input.shape}")
    # out = model(rgb_input, rf_input)
    # print(f"out: {out.shape}")
    
    # # 测试RGB_Model
    # rgb_input = torch.rand(size=(2, 3, 100, 32, 32)).to(device)
    # model = RGBModel(frame_depth=4, img_size=rgb_input.shape[-1], time_length=rgb_input.shape[2]).to(device)
    # bvp = model(rgb_input)
    # print(f"{bvp.shape}")
    
    # 测试RF_Model
    rf_input = torch.rand(size=(10, 4, 64, 256)).to(device)
    model = RFModel(time_length=rf_input.shape[2], freq_size=rf_input.shape[-1]).to(device)
    bvp = model(rf_input)
    print(f"{bvp.shape}")