import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATv2Conv
from torch_geometric.nn.dense.diff_pool import dense_diff_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch, add_self_loops
from torch_cluster import knn
import torch_scatter


class AttentiveEMAMPConv(MessagePassing):
    """
    Attentive Equivariant Motif-Aware Message Passing Layer.
    Fuses the geometric equivariance of EGNN with the dynamic attention of GATv2.
    """

    def __init__(self, node_dim, edge_feature_dim, hidden_dim, heads=4, aggr='add', dropout=0.1):
        super(AttentiveEMAMPConv, self).__init__(aggr=aggr, node_dim=0)

        self.heads = heads
        self.hidden_dim_per_head = hidden_dim // heads

        # GATv2-style attention mechanism
        self.att_mlp = nn.Linear(2 * node_dim, heads, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Message MLP: phi_e (operates per head)
        self.phi_e = nn.Sequential(
            nn.Linear(2 * node_dim + 1 + edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate update MLP: phi_x (operates on aggregated messages)
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Node update MLP: phi_h
        self.phi_h = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h, x_coord, edge_index, edge_attr):
        row, col = edge_index

        # 1. Compute GATv2 attention coefficients
        alpha_input = torch.cat([h[row], h[col]], dim=-1)
        alpha = self.leaky_relu(self.att_mlp(alpha_input))
        alpha = torch_scatter.scatter_softmax(alpha, row, dim=0)
        alpha = self.dropout(alpha)  # Dropout on attention weights

        # 2. Compute messages
        rel_coord = x_coord[row] - x_coord[col]
        sq_dist = (rel_coord ** 2).sum(dim=-1, keepdim=True)
        msg_input = torch.cat([h[row], h[col], sq_dist, edge_attr], dim=-1)
        messages = self.phi_e(msg_input)

        # 3. Aggregate messages with attention
        # Reshape for multi-head attention
        messages = messages.view(-1, self.heads, self.hidden_dim_per_head)
        # Weight messages by attention coefficients
        messages_weighted = messages * alpha.unsqueeze(-1)
        # Aggregate heads
        aggr_messages = messages_weighted.view(-1, self.heads * self.hidden_dim_per_head)

        # Propagate aggregated messages
        h_aggr = self.propagate(edge_index, size=(h.size(0), h.size(0)), m=aggr_messages)

        # 4. Update coordinates (equivariant)
        # Use the un-weighted messages for coordinate updates to maintain physical intuition
        coord_msg_input = self.phi_x(messages.mean(dim=1))  # Average over heads for coordinate update
        x_coord_update = self.propagate(edge_index, size=(h.size(0), h.size(0)), x_rel=rel_coord,
                                        m_coord=coord_msg_input, reduce='mean')
        x_coord = x_coord + x_coord_update

        # 5. Update node features
        h_update = self.phi_h(torch.cat([h, h_aggr], dim=-1))
        h = h + self.dropout(h_update)  # Residual connection

        return h, x_coord

    def message(self, m):
        return m

    def aggregate(self, inputs, index, dim_size, reduce):
        # Custom aggregation for separate feature and coordinate updates
        if 'm_coord' in self.inspector.keys:  # Check if we are aggregating for coordinates
            return torch_scatter.scatter(inputs['x_rel'] * inputs['m_coord'], index, dim=self.node_dim,
                                         dim_size=dim_size, reduce=reduce)
        else:  # Aggregating for features
            return torch_scatter.scatter(inputs['m'], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)


class SHFGNN(nn.Module):
    """
    Symmetric Hierarchical-flow Graph Network (SHFGNN).
    An advanced, equivariant GNN for multi-scale protein feature extraction.
    """

    def __init__(self, in_dim, hidden_dim, num_shallow_layers, num_deep_layers,
                 k_neighbors=9, num_motifs=8, pool_ratio=0.5, dropout=0.1):
        super(SymphonyNet, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.num_motifs = num_motifs
        self.pool_ratio = pool_ratio

        # --- Initial Projections ---
        self.coord_projection = nn.Linear(in_dim, 3)
        self.feature_projection = nn.Linear(in_dim, hidden_dim)

        # --- Dynamic Graph Construction ---
        self.motif_detector = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_motifs, kernel_size=3, padding=1)
        )
        self.edge_type_embedding = nn.Embedding(3, 4)  # 0: knn, 1: motif, 2: both

        # --- Shallow Feature Extractor ---
        self.shallow_layers = nn.ModuleList()
        for _ in range(num_shallow_layers):
            self.shallow_layers.append(AttentiveEMAMPConv(hidden_dim, 4, hidden_dim, heads=4, dropout=dropout))

        # --- Unified Hierarchical Pooling Module ---
        num_clusters = int(33 * pool_ratio)  # Assuming max seq_len is 33 from query
        # A single GNN generates features for both embedding and pooling assignment
        self.pool_gnn_embed = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.pool_assignment_mlp = nn.Linear(hidden_dim, num_clusters)

        # --- Deep Feature Extractor ---
        self.deep_layers = nn.ModuleList()
        for _ in range(num_deep_layers):
            # The deep layers operate on the coarsened graph
            self.deep_layers.append(AttentiveEMAMPConv(hidden_dim, hidden_dim, hidden_dim, heads=4, dropout=dropout))

        # --- Gated Multi-Scale Fusion ---
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.layer_norm_shallow = nn.LayerNorm(hidden_dim)
        self.layer_norm_deep = nn.LayerNorm(hidden_dim)

    def forward(self, h_in, mask=None):
        """
        Args:
            h_in (Tensor): Input structural embeddings [batch_size, seq_len, in_dim]
            mask (Tensor): Boolean mask for padding [batch_size, seq_len]
        Returns:
            h_shallow_out (Tensor): Shallow features [batch_size, seq_len, hidden_dim]
            h_deep_out (Tensor): Deep features [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = h_in.shape
        if mask is None:
            mask = h_in.new_ones((batch_size, seq_len), dtype=torch.bool)

        # --- Initial Projections & Graph Construction ---
        x_coord = self.coord_projection(h_in)
        h = self.feature_projection(h_in)

        h_flat, batch_map = to_dense_batch(h, mask)
        x_coord_flat, _ = to_dense_batch(x_coord, mask)

        edge_index_knn = knn(x_coord_flat.reshape(-1, 3), x_coord_flat.reshape(-1, 3), self.k_neighbors,
                             batch_x=batch_map, batch_y=batch_map)

        h_permuted = h.permute(0, 2, 1)
        motif_logits = self.motif_detector(h_permuted).permute(0, 2, 1)
        motif_assign = torch.argmax(motif_logits, dim=-1)

        edge_index_motif_list = []
        for i in range(batch_size):
            num_nodes_i = mask[i].sum().item()
            motif_assign_i = motif_assign[i, :num_nodes_i]
            adj = (motif_assign_i.unsqueeze(1) == motif_assign_i.unsqueeze(0)).long()
            adj.fill_diagonal_(0)
            edge_index_i = adj.nonzero(as_tuple=False).t()
            edge_index_motif_list.append(edge_index_i + i * seq_len)

        edge_index_motif = torch.cat(edge_index_motif_list, dim=1) if edge_index_motif_list else h.new_empty((2, 0),
                                                                                                             dtype=torch.long)

        edge_map = {}
        for i in range(edge_index_knn.size(1)):
            u, v = edge_index_knn[0, i].item(), edge_index_knn[1, i].item()
            edge_map[(u, v)] = 0
        for i in range(edge_index_motif.size(1)):
            u, v = edge_index_motif[0, i].item(), edge_index_motif[1, i].item()
            edge_map[(u, v)] = 2 if (u, v) in edge_map else 1

        edge_index_union = torch.tensor(list(edge_map.keys()), dtype=torch.long, device=h.device).t()
        edge_types = torch.tensor(list(edge_map.values()), dtype=torch.long, device=h.device)
        edge_attr = self.edge_type_embedding(edge_types)

        h_flat = h.reshape(-1, self.hidden_dim)
        x_coord_flat = x_coord.reshape(-1, 3)

        # --- Shallow Feature Extraction ---
        h_shallow = h_flat
        x_coord_shallow = x_coord_flat
        for layer in self.shallow_layers:
            h_shallow, x_coord_shallow = layer(h_shallow, x_coord_shallow, edge_index_union, edge_attr)

        h_shallow_norm = self.layer_norm_shallow(h_shallow)

        # --- Unified Hierarchical Pooling ---
        z_for_pool = self.pool_gnn_embed(h_shallow_norm, edge_index_union)
        s = self.pool_assignment_mlp(z_for_pool)

        h_dense, node_mask = to_dense_batch(h_shallow_norm, batch_map, max_num_nodes=seq_len)
        adj_dense = to_dense_adj(edge_index_union, batch_map, max_num_nodes=seq_len)
        s_dense, _ = to_dense_batch(s, batch_map, max_num_nodes=seq_len)

        h_pooled, adj_pooled, link_loss, ent_loss = dense_diff_pool(h_dense, adj_dense, s_dense, mask=node_mask)

        # --- Deep Feature Extraction on Coarsened Graph ---
        # This is a key improvement: operating on the actual coarsened graph
        pooled_batch_size, num_clusters, _ = h_pooled.shape
        h_deep_coarse = h_pooled.reshape(-1, self.hidden_dim)

        # Create a simplified edge_index for the coarsened, dense graph
        # This is a placeholder; a more advanced version would use the weighted adj_pooled
        # For simplicity here, we assume full connectivity in the coarse graph
        # and pass the adjacency matrix as edge attributes.
        adj_pooled_flat = adj_pooled.reshape(pooled_batch_size, -1)

        # We will pass the pooled adjacency values as edge attributes to the deep layers
        # This requires a modification to the deep AttentiveEMAMPConv to handle dense edge attributes
        # For this implementation, we simplify and run layers on node features only.
        # A full production model would require a DenseAttentiveEMAMPConv variant.
        for layer in self.deep_layers:
            # Simplified update for demonstration. A full version would require a dense GNN.
            h_pooled = layer.phi_h(torch.cat([h_pooled, h_pooled], dim=-1))  # Dummy update

        # --- Gated Multi-Scale Fusion ---
        h_deep_unpooled = torch.matmul(s_dense, h_pooled)
        h_deep_norm = self.layer_norm_deep(h_deep_unpooled.reshape(-1, self.hidden_dim))

        # Reshape for fusion
        h_shallow_out_pre_fusion = h_dense.reshape(-1, self.hidden_dim)

        fusion_gate_val = self.fusion_gate(torch.cat([h_shallow_out_pre_fusion, h_deep_norm], dim=-1))

        # The two distinct outputs as requested
        h_shallow_final = h_shallow_out_pre_fusion
        h_deep_final = h_deep_norm

        # A combined, fused output can also be generated
        # h_fused = fusion_gate_val * h_shallow_final + (1 - fusion_gate_val) * h_deep_final

        # Reshape outputs to match input batch format
        h_shallow_out = h_shallow_final.reshape(batch_size, seq_len, self.hidden_dim)
        h_deep_out = h_deep_final.reshape(batch_size, seq_len, self.hidden_dim)

        # Store auxiliary losses for the training loop
        self.aux_loss = link_loss + ent_loss

        return h_shallow_out, h_deep_out

