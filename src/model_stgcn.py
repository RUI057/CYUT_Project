"""ST-GCN 手語辨識模型（Spatial-Temporal Graph Convolutional Network）。

與 GestureLSTM 介面一致：forward 吃 (B, T, 126)、輸出 (B, num_classes)，
內部把 126 維 reshape 成圖結構 (B, C=3, T, V=21, M=2 隻手)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph import hand_adjacency, NUM_NODE


class GraphConv(nn.Module):
    """空間圖卷積：對 K 個分區各做 1x1 conv，再用鄰接矩陣聚合。"""
    def __init__(self, in_c, out_c, K):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(in_c, out_c * K, kernel_size=1)

    def forward(self, x, A):                    # x:(N,C,T,V)  A:(K,V,V)
        x = self.conv(x)
        N, KC, T, V = x.size()
        x = x.view(N, self.K, KC // self.K, T, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        return x.contiguous()


class STGCNBlock(nn.Module):
    """空間圖卷積 + 時間卷積（kernel=9）+ 殘差。"""
    def __init__(self, in_c, out_c, K, stride=1, residual=True, dropout=0.3):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c, K)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_c == out_c and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, (stride, 1)),
                nn.BatchNorm2d(out_c),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)


class GestureSTGCN(nn.Module):
    def __init__(self, feat_dim=126, num_classes=10, in_channels=3,
                 num_point=NUM_NODE, num_person=2, dropout=0.3, edge_importance=True):
        super().__init__()
        A = torch.tensor(hand_adjacency(), dtype=torch.float32)  # (K,V,V)
        self.register_buffer("A", A)
        K = A.size(0)

        self.num_point   = num_point
        self.num_person  = num_person
        self.in_channels = in_channels
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64,  K, residual=False),
            STGCNBlock(64,  64,  K),
            STGCNBlock(64,  128, K, stride=2),
            STGCNBlock(128, 256, K, stride=2),
        ])
        # 可學習的邊重要度（讓模型自行加權拓樸連結）
        if edge_importance:
            self.edge_imp = nn.ParameterList(
                [nn.Parameter(torch.ones_like(A)) for _ in self.blocks])
        else:
            self.edge_imp = [1] * len(self.blocks)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):                        # x: (B, T, 126)
        B, T, _ = x.size()
        # (B,T,126) → (B, C=3, T, V=21, M=2)；126 排列為 [左手63][右手63]，每手 21×(x,y,z)
        x = x.view(B, T, self.num_person, self.num_point, self.in_channels)
        x = x.permute(0, 4, 1, 3, 2).contiguous()           # (B,C,T,V,M)
        N, C, T, V, M = x.size()

        # data batch-norm（攤平成 (N, M*V*C, T)）
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)               # 每隻手當一個樣本進骨幹

        for blk, imp in zip(self.blocks, self.edge_imp):
            x = blk(x, self.A * imp)

        x = F.avg_pool2d(x, x.size()[2:])        # 全域池化 → (N*M, 256, 1, 1)
        x = x.view(N, M, -1).mean(dim=1)         # 兩隻手平均 → (N, 256)
        return self.fc(x)
