# coding: utf-8
r"""
NTCF: Neural Tree Collaborative Filtering with Hierarchical Message Passing
################################################################################
    
Key Innovation:
    Tree-structured message passing where each node can serve as a root with 
    personalized propagation depth based on node centrality in the interaction graph.
    When all nodes have uniform depth, NTCF degenerates to NGCF (lower bound).
    
Theoretical Foundation:
    - Tree as special graph: any node can be root, children = interacted nodes
    - Bidirectionality preserved through different root perspectives
    - Hierarchical propagation enables node-specific personalization
    - Full interaction graph preserved (no information loss)
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole.model.init import xavier_normal_initialization

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import BiGNNConv


# ntcf.py 重构版 - Bakry-Emery 曲率版本
class NTCF(GeneralGraphRecommender):
    """
    NTCF: Neural Tree Collaborative Filtering with Bakry-Emery Curvature
    Optimized version based on NGCF architecture with curvature-based adaptive layer propagation
    
    Key Innovation:
        - Uses Bakry-Emery curvature to determine propagation depth per node
        - Positive curvature (popular items): propagate fewer layers (layer - 1)
        - Negative curvature (niche items): propagate more layers (layer + 1)
        - Near-zero curvature (normal): propagate standard layers
    """
    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset):
        super(NTCF, self).__init__(config, dataset)
        
        # 超参数（保持与 NGCF 一致）
        self.embedding_size = config['embedding_size']
        
        # 处理 n_layers 配置
        n_layers_config = config['n_layers']
        self.n_layers = n_layers_config[0] if isinstance(n_layers_config, list) else n_layers_config
        
        self.reg_weight = config['reg_weight']
        self.mess_dropout = config['mess_dropout']
        
        # 曲率相关参数
        self.curvature_strategy = config['curvature_strategy']
        self.curvature_threshold = config['curvature_threshold']  # 判断为"均匀"的阈值范围
        
        # 聚合策略
        self.aggregation_type = config['aggregation_type']
        
        # 加载交互矩阵
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # 初始化嵌入
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
        # 【关键改进】使用 BiGNNConv（与 NGCF 相同）
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gnn_layers.append(BiGNNConv(self.embedding_size, self.embedding_size))
        
        # 损失函数
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        
        # 缓存变量
        self.restore_user_e = None
        self.restore_item_e = None
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']
        
        # 预处理缓存
        self._preprocessing_cache = {}
        
        # 曲率缓存
        self._curvature_cache = {}
        
        # 初始化
        self.apply(xavier_normal_initialization)
    
    def prepare_preprocessing(self):
        """
        计算节点曲率和层调整（只在第一个 epoch 前执行一次）
        """
        if self._preprocessing_cache.get('is_ready', False):
            return
        
        # 计算节点曲率
        curvatures = self.calculate_bakry_emery_curvatures()
        node_curvatures = torch.tensor(curvatures, dtype=torch.float32).to(self.device)
        
        # 根据曲率计算每层的有效传播掩码
        # 曲率 -> 层调整：负曲率 (layer+1), 正曲率 (layer-1), 均匀 (layer)
        layer_adjustments = self._compute_layer_adjustments(curvatures)
        node_layer_adjustments = torch.tensor(layer_adjustments, dtype=torch.long).to(self.device)
        
        # 预计算每层的 mask [n_layers, n_nodes, 1]
        # mask[layer][node] = 1 表示该节点在该层应该更新，否则保持原值
        layer_masks = []
        for layer in range(self.n_layers):
            # 节点的最终传播层数 = n_layers + layer_adjustment
            # 如果 layer < 最终层数，则该节点需要传播
            effective_layers = self.n_layers + node_layer_adjustments
            mask = (effective_layers > layer).float().unsqueeze(1)
            layer_masks.append(mask)
        
        # 堆叠所有层：[n_layers, n_nodes, 1]
        layer_masks = torch.stack(layer_masks, dim=0)
        
        # 缓存
        self._preprocessing_cache['node_curvatures'] = node_curvatures
        self._preprocessing_cache['node_layer_adjustments'] = node_layer_adjustments
        self._preprocessing_cache['layer_masks'] = layer_masks
        self._preprocessing_cache['is_ready'] = True
        
        # 显示曲率统计信息
        self._log_curvature_statistics(curvatures, layer_adjustments)
    
    def _compute_layer_adjustments(self, curvatures):
        """
        根据曲率计算层调整量
        负曲率 -> layer + 1 (需要更多传播)
        正曲率 -> layer - 1 (需要较少传播)
        均匀曲率 -> layer (正常传播)
        """
        n_nodes = len(curvatures)
        adjustments = np.zeros(n_nodes, dtype=np.int64)
        
        # 使用阈值范围判断均匀曲率
        for i, curvature in enumerate(curvatures):
            if curvature < -self.curvature_threshold:
                # 负曲率：冷门商品，需要更多传播
                adjustments[i] = 1
            elif curvature > self.curvature_threshold:
                # 正曲率：热门商品，需要较少传播
                adjustments[i] = -1
            else:
                # 均匀曲率：正常传播
                adjustments[i] = 0
        
        return adjustments
    
    def _log_curvature_statistics(self, curvatures, layer_adjustments):
        """
        记录并显示节点曲率统计信息
        """
        import logging
        logger = logging.getLogger()
        
        n_nodes = len(curvatures)
        n_users = self.n_users
        n_items = self.n_items
        
        # 曲率统计
        min_curvature = curvatures.min()
        max_curvature = curvatures.max()
        mean_curvature = curvatures.mean()
        median_curvature = np.median(curvatures)
        std_curvature = curvatures.std()
        
        # 曲率分布
        negative_mask = curvatures < -self.curvature_threshold
        positive_mask = curvatures > self.curvature_threshold
        uniform_mask = ~negative_mask & ~positive_mask
        
        n_negative = negative_mask.sum()
        n_positive = positive_mask.sum()
        n_uniform = uniform_mask.sum()
        
        pct_negative = n_negative / n_nodes * 100
        pct_positive = n_positive / n_nodes * 100
        pct_uniform = n_uniform / n_nodes * 100
        
        # 层调整统计
        adjust_counts = np.bincount(layer_adjustments + 1, minlength=3)  # +1 to make indices non-negative
        adjust_labels = ['layer+1 (negative)', 'layer (uniform)', 'layer-1 (positive)']
        
        # 用户和物品分别统计
        user_curvatures = curvatures[:n_users]
        item_curvatures = curvatures[n_users:]
        
        user_mean = user_curvatures.mean()
        item_mean = item_curvatures.mean()
        
        user_negative = (user_curvatures < -self.curvature_threshold).sum()
        user_positive = (user_curvatures > self.curvature_threshold).sum()
        item_negative = (item_curvatures < -self.curvature_threshold).sum()
        item_positive = (item_curvatures > self.curvature_threshold).sum()
        
        # 显示信息
        logger.info("=" * 60)
        logger.info("NTCF Bakry-Emery Curvature Statistics")
        logger.info("=" * 60)
        logger.info(f"Curvature Strategy: {self.curvature_strategy}")
        logger.info(f"Curvature Threshold: ±{self.curvature_threshold}")
        logger.info(f"Total Nodes: {n_nodes:,} (Users: {n_users:,}, Items: {n_items:,})")
        logger.info("-" * 60)
        logger.info("Overall Curvature Distribution:")
        logger.info(f"  Min Curvature: {min_curvature:.6f}")
        logger.info(f"  Max Curvature: {max_curvature:.6f}")
        logger.info(f"  Mean Curvature: {mean_curvature:.6f}")
        logger.info(f"  Median Curvature: {median_curvature:.6f}")
        logger.info(f"  Std Curvature: {std_curvature:.6f}")
        logger.info("-" * 60)
        logger.info("Curvature-based Layer Adjustment:")
        logger.info(f"  Negative Curvature (<-{self.curvature_threshold}): {n_negative:,} nodes ({pct_negative:.2f}%) -> layer + 1")
        logger.info(f"  Uniform Curvature (±{self.curvature_threshold}): {n_uniform:,} nodes ({pct_uniform:.2f}%) -> layer")
        logger.info(f"  Positive Curvature (>{self.curvature_threshold}): {n_positive:,} nodes ({pct_positive:.2f}%) -> layer - 1")
        logger.info("-" * 60)
        logger.info("Layer Adjustment Distribution:")
        for i, (count, label) in enumerate(zip(adjust_counts, adjust_labels)):
            pct = count / n_nodes * 100
            logger.info(f"  {label}: {count:,} nodes ({pct:.2f}%)")
        logger.info("-" * 60)
        logger.info("User vs Item Curvature Comparison:")
        logger.info(f"  User Mean Curvature: {user_mean:.6f} (negative: {user_negative}, positive: {user_positive})")
        logger.info(f"  Item Mean Curvature: {item_mean:.6f} (negative: {item_negative}, positive: {item_positive})")
        logger.info(f"  Difference: {abs(user_mean - item_mean):.6f}")
        logger.info("=" * 60)
    
    def calculate_bakry_emery_curvatures(self):
        """
        计算每个节点的 Bakry-Emery 曲率
        
        Bakry-Emery 曲率基于图的局部结构和节点连接性。
        在推荐系统中：
        - 热门商品（高度数）：正曲率，信息容易饱和，传播层数应减少
        - 冷门商品（低度数）：负曲率，信息稀疏，传播层数应增加
        - 普通商品：曲率接近 0，正常传播
        
        曲率计算公式（简化版）：
        curvature(node) = (avg_neighbor_degree - node_degree) / (avg_neighbor_degree + node_degree + epsilon)
        
        这个公式捕捉了节点与其邻居的相对连接性：
        - 如果 node_degree > avg_neighbor_degree: 负曲率（节点比邻居更孤立）
        - 如果 node_degree < avg_neighbor_degree: 正曲率（节点比邻居更中心）
        - 如果接近：曲率接近 0
        """
        n_nodes = self.n_users + self.n_items
        curvatures = np.zeros(n_nodes)
        
        # 转换为 CSR 格式以便高效行索引
        interaction_csr = self.interaction_matrix.tocsr()
        # 转换为 CSC 格式以便高效列索引
        interaction_csc = self.interaction_matrix.tocsc()
        
        # 计算所有节点的度数
        degrees = np.zeros(n_nodes)
        user_degrees = np.array(interaction_csr.sum(axis=1)).flatten()
        degrees[:self.n_users] = user_degrees
        item_degrees = np.array(interaction_csr.sum(axis=0)).flatten()
        degrees[self.n_users:] = item_degrees
        
        # 对每个节点计算曲率
        epsilon = 1e-8  # 避免除零
        
        for node_id in range(n_nodes):
            node_degree = degrees[node_id]
            
            if node_degree == 0:
                # 孤立节点，曲率为 0
                curvatures[node_id] = 0.0
                continue
            
            # 获取邻居
            if node_id < self.n_users:
                # 用户节点：邻居是物品（行索引）
                neighbors = interaction_csr[node_id].nonzero()[1]
            else:
                # 物品节点：邻居是用户（列索引）
                item_idx = node_id - self.n_users
                neighbors = interaction_csc[:, item_idx].nonzero()[0]
            
            if len(neighbors) == 0:
                curvatures[node_id] = 0.0
                continue
            
            # 计算邻居的平均度数
            neighbor_degrees = degrees[neighbors]
            avg_neighbor_degree = neighbor_degrees.mean()
            
            # Bakry-Emery 曲率公式（简化版）
            # 正曲率：node 比邻居更中心（度数更低但连接紧密）
            # 负曲率：node 比邻居更边缘（度数更高或更孤立）
            curvature = (avg_neighbor_degree - node_degree) / (avg_neighbor_degree + node_degree + epsilon)
            
            curvatures[node_id] = curvature
        
        return curvatures
    
    def _compute_pagerank_scores(self):
        """
        计算 PageRank 分数（保留用于兼容性）
        """
        n_nodes = self.n_users + self.n_items
        # 简化的 PageRank 计算
        damping_factor = 0.85
        max_iter = 100
        tol = 1e-6
        
        # 初始化
        scores = np.ones(n_nodes) / n_nodes
        
        # 构建转移矩阵
        row, col = self.interaction_matrix.nonzero()
        # 双向图
        row = np.concatenate([row, col + self.n_users])
        col = np.concatenate([col + self.n_users, row[:len(row)//2]])
        
        degrees = np.bincount(row, minlength=n_nodes)
        degrees = np.maximum(degrees, 1)  # 避免除零
        
        for _ in range(max_iter):
            new_scores = (1 - damping_factor) / n_nodes + damping_factor * np.bincount(
                col, weights=scores[row] / degrees[row], minlength=n_nodes
            )
            
            if np.abs(new_scores - scores).sum() < tol:
                break
            scores = new_scores
        
        return scores
    
    def _compute_ego_network_size(self):
        """
        计算 ego network 大小（保留用于兼容性）
        """
        n_nodes = self.n_users + self.n_items
        sizes = np.zeros(n_nodes)
        
        # 转换为 CSR 和 CSC 格式以支持索引
        interaction_csr = self.interaction_matrix.tocsr()
        interaction_csc = self.interaction_matrix.tocsc()
        
        for node_id in range(n_nodes):
            if node_id < self.n_users:
                neighbors = interaction_csr[node_id].nonzero()[1]
            else:
                item_idx = node_id - self.n_users
                neighbors = interaction_csc[:, item_idx].nonzero()[0]
            
            sizes[node_id] = len(neighbors)
        
        return sizes
    
    def forward(self, return_intermediate=False):
        """
        前向传播：使用 BiGNNConv + 曲率基础层调整机制
        """
        # 确保预处理已完成
        if not self._preprocessing_cache.get('is_ready', False):
            self.prepare_preprocessing()
        
        # 初始嵌入
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]

        all_embeddings_layer = ego_embeddings
        
        # 获取预计算的层掩码
        layer_masks = self._preprocessing_cache['layer_masks']
        
        # 【关键改进】使用 BiGNNConv 进行传播，基于曲率调整每层传播
        for layer_idx, gnn_layer in enumerate(self.gnn_layers):
            # 1. BiGNNConv 传播（高效，使用稀疏矩阵乘法）
            all_embeddings_layer = gnn_layer(all_embeddings_layer, self.edge_index, self.edge_weight)
            
            # 2. 激活函数
            all_embeddings_layer = nn.LeakyReLU(negative_slope=0.2)(all_embeddings_layer)
            
            # 3. Dropout
            all_embeddings_layer = nn.Dropout(self.mess_dropout)(all_embeddings_layer)
            
            # 4. 归一化
            all_embeddings_layer = F.normalize(all_embeddings_layer, p=2, dim=1)
            
            # 5. 【NTCF 核心】应用曲率基础层掩码
            # 使用乘法代替 torch.where（更快）
            active_mask = layer_masks[layer_idx]  # [n_nodes, 1]
            all_embeddings_layer = active_mask * all_embeddings_layer + (1 - active_mask) * ego_embeddings
            
            # 6. 保存当前层嵌入
            all_embeddings.append(all_embeddings_layer)
        
        # 聚合策略
        if self.aggregation_type == 'concat':
            # NGCF 风格：拼接所有层
            final_embeddings = torch.cat(all_embeddings, dim=1)
        else:
            # 默认：层间均值聚合
            final_embeddings = self._layer_mean_aggregate(all_embeddings)
        
        # 分割用户和物品
        user_all_embeddings, item_all_embeddings = torch.split(
            final_embeddings, [self.n_users, self.n_items], dim=0
        )
        
        if return_intermediate:
            return user_all_embeddings, item_all_embeddings, all_embeddings
        else:
            return user_all_embeddings, item_all_embeddings
    
    def _layer_mean_aggregate(self, all_embeddings):
        """
        层间均值聚合（基于曲率调整）
        """
        n_embeddings = len(all_embeddings)
        node_layer_adjustments = self._preprocessing_cache['node_layer_adjustments']
        
        # 计算每个节点的实际传播层数
        # effective_layers = n_layers + adjustment
        effective_layers = self.n_layers + node_layer_adjustments
        
        # 堆叠所有层：[n_layers+1, n_nodes, dim]
        stacked_embeddings = torch.stack(all_embeddings, dim=0)
        
        # 创建累积 mask
        layer_indices = torch.arange(n_embeddings, device=self.device).view(-1, 1, 1)
        effective_layers_expanded = effective_layers.view(1, -1, 1)
        mask = (layer_indices <= effective_layers_expanded).float()
        
        # 加权求和
        masked_embeddings = stacked_embeddings * mask
        summed_embeddings = torch.sum(masked_embeddings, dim=0)
        
        # 归一化（每个节点的有效层数 +1，因为包含第 0 层）
        counts = (effective_layers + 1).view(-1, 1).float()
        final_embeddings = summed_embeddings / counts
        
        return final_embeddings
    
    def calculate_loss(self, interaction):
        # 清除缓存
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        
        # 获取索引
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        
        # 前向传播
        user_all_embeddings, item_all_embeddings = self.forward()
        
        # 获取批次嵌入
        u_embeddings = user_all_embeddings[users]
        pos_embeddings = item_all_embeddings[pos_items]
        neg_embeddings = item_all_embeddings[neg_items]
        
        # BPR 损失
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        
        # 正则化
        u_ego_embeddings = self.user_embedding(users)
        pos_ego_embeddings = self.item_embedding(pos_items)
        neg_ego_embeddings = self.item_embedding(neg_items)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        
        return mf_loss + self.reg_weight * reg_loss
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)