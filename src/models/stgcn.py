"""
ST-GCN: Spatial-Temporal Graph Convolutional Network para LSP.
Opera sobre secuencias de landmarks MediaPipe [T, N_kp, 3].

Referencia: Yan et al., 2018 — "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple


class GraphConv(nn.Module):
    """Convolución espectral sobre grafo con adyacencia aprendible."""

    def __init__(self, in_channels: int, out_channels: int, n_nodes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * 3, kernel_size=1)
        # Adyacencia aprendible (residual sobre la fija)
        self.A_learnable = nn.Parameter(torch.zeros(3, n_nodes, n_nodes))
        nn.init.normal_(self.A_learnable, 0, 0.01)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C, T, N]
        A : [3, N, N]   adyacencia fija (particionada)
        """
        # Añadir componente aprendible
        A = A + self.A_learnable

        # Convolución sobre canales → 3 particiones
        x = self.conv(x)                              # [B, 3*out, T, N]
        B, C3, T, N = x.shape
        out_ch = C3 // 3
        x = x.view(B, 3, out_ch, T, N)               # [B, 3, out, T, N]

        # Multiplicación por adyacencia (mensaje passing)
        # y[b,c,t,n] = sum_k sum_m A[k,n,m] * x[b,k,c,t,m]
        # Implementación eficiente via bmm para evitar OOM en MPS/CUDA
        # x: [B, 3, C, T, N] → [3, B*C*T, N]
        B2, K, C2, T2, N2 = x.shape
        x_r = x.permute(1, 0, 2, 3, 4).reshape(K, B2 * C2 * T2, N2)  # [3, BCT, N]
        y = torch.bmm(x_r, A.transpose(1, 2))                          # [3, BCT, N]
        y = y.reshape(K, B2, C2, T2, N2).sum(0)                        # [B, C, T, N]

        return y


class STGCNBlock(nn.Module):
    """Bloque básico ST-GCN: GCN espacial + convolución temporal."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_nodes: int,
        temporal_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gcn = GraphConv(in_channels, out_channels, n_nodes)
        self.bn_gcn = nn.BatchNorm2d(out_channels)

        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1),
                      padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        # Conexión residual
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.relu(self.bn_gcn(self.gcn(x, A)))
        x = self.tcn(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    """
    ST-GCN completo para clasificación de señas LSP.

    Input : landmarks [B, C_in, T, N]   — típicamente C_in=3 (x,y,z), T=30, N=75
    Output: logits [B, n_classes]

    N = 42 (manos) + 33 (pose) = 75 keypoints por defecto.
    """

    # Adyacencia anatómica de manos (21 nodos cada una)
    HAND_EDGES = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    # Adyacencia de pose (upper body relevante)
    POSE_EDGES = [
        (11,12),(11,13),(13,15),(12,14),(14,16),  # brazos
        (11,23),(12,24),(23,24),                   # tronco
        (15,17),(15,19),(16,18),(16,20),           # manos/muñecas
    ]

    def __init__(
        self,
        n_classes: int,
        n_nodes: int = 75,         # 42 manos + 33 pose
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.3,
        temporal_kernel: int = 9,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_classes = n_classes

        # Construir matriz de adyacencia
        A = self._build_adjacency(n_nodes)
        self.register_buffer('A', A)

        # Normalización de entrada
        self.input_bn = nn.BatchNorm1d(in_channels * n_nodes)

        # Proyección inicial
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # Bloques ST-GCN (soporta hasta 8 capas)
        channels = [hidden_channels, hidden_channels, hidden_channels * 2,
                    hidden_channels * 2, hidden_channels * 4,
                    hidden_channels * 4, hidden_channels * 4, hidden_channels * 4,
                    hidden_channels * 4]
        strides  = [1, 1, 2, 1, 2, 1, 1, 1]

        layers = []
        for i in range(num_layers):
            in_ch  = channels[i]
            out_ch = channels[i + 1]
            layers.append(STGCNBlock(
                in_ch, out_ch, n_nodes,
                temporal_kernel=temporal_kernel,
                stride=strides[i],
                dropout=dropout,
            ))
        self.stgcn_layers = nn.ModuleList(layers)

        # Clasificador
        final_ch = channels[num_layers]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_ch, n_classes),
        )

    def _build_adjacency(self, n_nodes: int) -> torch.Tensor:
        """Construye A particionada en 3 subsets: identidad, vecindad cercana, lejana."""
        A = np.zeros((3, n_nodes, n_nodes), dtype=np.float32)

        # Subset 0: autoconexión
        np.fill_diagonal(A[0], 1)

        def add_edges(edges, offset_a: int = 0, offset_b: int = 0):
            for a, b in edges:
                A[1, offset_a + a, offset_b + b] = 1
                A[1, offset_b + b, offset_a + a] = 1

        # Mano izquierda (nodos 0–20)
        add_edges(self.HAND_EDGES, 0, 0)
        # Mano derecha (nodos 21–41)
        add_edges(self.HAND_EDGES, 21, 21)
        # Pose (nodos 42–74)
        if n_nodes > 42:
            add_edges(self.POSE_EDGES, 42, 42)
            # Conexión muñeca pose → punta mano
            A[2, 15, 0]  = 1   # muñeca pose izq → mano izq
            A[2, 16, 21] = 1   # muñeca pose der → mano der

        # Normalización por grado
        for k in range(3):
            D = A[k].sum(axis=1)
            D = np.where(D > 0, 1.0 / D, 0.0)
            A[k] = np.diag(D) @ A[k]

        return torch.from_numpy(A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: landmarks [B, T, N, 3]  → se reordena internamente a [B, 3, T, N]
        """
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()   # [B, 3, T, N]

        B, C, T, N = x.shape

        # Normalización de entrada
        x_flat = x.permute(0, 2, 3, 1).reshape(B * T, N * C)
        # BN sobre toda la secuencia no es lo ideal pero funciona como proxy
        x = x.permute(0, 2, 1, 3).reshape(B * T, C, 1, N)
        x = x.reshape(B, C, T, N)

        x = self.input_proj(x)    # [B, hidden, T, N]

        for layer in self.stgcn_layers:
            x = layer(x, self.A)

        x = self.pool(x)          # [B, final_ch, 1, 1]
        x = x.flatten(1)          # [B, final_ch]
        return self.classifier(x) # [B, n_classes]

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.input_proj(x)
        for layer in self.stgcn_layers:
            x = layer(x, self.A)
        x = self.pool(x).flatten(1)
        return x
