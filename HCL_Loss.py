class HierarchicalContrastiveLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha  # 跨层对比权重
        
    def forward(self, projections, labels):
        z1, z2, z3 = projections
        batch_size = z1.size(0)
        
        # 生成标签掩码 (同一类为正样本)
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B,B]
        
        # 各层内部对比损失
        loss_h1 = self._contrastive_loss(z1, label_mask)
        loss_h2 = self._contrastive_loss(z2, label_mask)
        loss_h3 = self._contrastive_loss(z3, label_mask)
        
        # 跨层对比损失
        loss_cross = 0.5*(self._cross_layer_loss(z1,z2,label_mask) + 
                         self._cross_layer_loss(z2,z3,label_mask))
        
        total_loss = (loss_h1 + loss_h2 + loss_h3)/3 + self.alpha*loss_cross
        return total_loss

    def _contrastive_loss(self, z, mask):
        sim = torch.mm(z, z.T)  # [B,B]
        sim /= 0.1  # 温度系数
        
        pos = sim[mask].reshape(z.size(0), -1)
        neg = sim[~mask].reshape(z.size(0), -1)
        
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(z.size(0), dtype=torch.long).to(z.device)
        
        return F.cross_entropy(logits, labels)

    def _cross_layer_loss(self, z_a, z_b, mask):
        sim = torch.mm(z_a, z_b.T)
        sim /= 0.1
        
        pos = sim.diag().unsqueeze(1)  # 同一样本跨层特征为正对
        neg = sim[~torch.eye(z_a.size(0), dtype=bool)].view(z_a.size(0), -1)
        
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(z_a.size(0), dtype=torch.long).to(z_a.device)
        
        return F.cross_entropy(logits, labels)