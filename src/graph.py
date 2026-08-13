"""手部骨架圖：MediaPipe 21 點拓樸 + ST-GCN 空間分區鄰接矩陣。

採用 Yan et al. (2018) 的 spatial configuration partitioning：
依「到手腕(中心節點)的距離」把鄰居分成 root / 向心 / 離心三組，
回傳 (K=3, 21, 21) 的正規化鄰接矩陣堆疊。
"""
import numpy as np

NUM_NODE = 21
CENTER   = 0   # 手腕為運動學根節點（中心）

# MediaPipe 手部連線（骨頭）
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),          # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),     # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),   # 無名指
    (13, 17), (17, 18), (18, 19), (19, 20),  # 小指
    (0, 17),                                 # 手腕—小指根
]


def _hop_distance(num_node, edges, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    hop_dis = np.full((num_node, num_node), np.inf)
    transfer = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive = (np.stack(transfer) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive[d]] = d
    return hop_dis


def _normalize_digraph(A):
    """列正規化 A·D^-1，使每個節點聚合後尺度一致。"""
    Dl = A.sum(0)
    Dn = np.zeros_like(A)
    for i in range(len(A)):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return A @ Dn


def hand_adjacency(max_hop=1, dilation=1):
    """回傳 (K, 21, 21) 的空間分區鄰接矩陣（float32）。K=3（max_hop=1）。"""
    hop_dis   = _hop_distance(NUM_NODE, HAND_EDGES, max_hop)
    valid_hop = range(0, max_hop + 1, dilation)

    adjacency = np.zeros((NUM_NODE, NUM_NODE))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    norm_adj = _normalize_digraph(adjacency)

    A = []
    for hop in valid_hop:
        a_root, a_close, a_further = (np.zeros((NUM_NODE, NUM_NODE)) for _ in range(3))
        for i in range(NUM_NODE):
            for j in range(NUM_NODE):
                if hop_dis[j, i] == hop:
                    if hop_dis[j, CENTER] == hop_dis[i, CENTER]:
                        a_root[j, i] = norm_adj[j, i]      # 同層
                    elif hop_dis[j, CENTER] > hop_dis[i, CENTER]:
                        a_close[j, i] = norm_adj[j, i]     # 向心（鄰居較遠 → 指向較近的 i）
                    else:
                        a_further[j, i] = norm_adj[j, i]   # 離心
        if hop == 0:
            A.append(a_root)
        else:
            A.append(a_root + a_close)
            A.append(a_further)
    return np.stack(A).astype(np.float32)


if __name__ == "__main__":
    A = hand_adjacency()
    print("鄰接矩陣形狀:", A.shape)          # (3, 21, 21)
    print("每個分區的連結數:", [int((a > 0).sum()) for a in A])
    print("拓樸邊數:", len(HAND_EDGES))
