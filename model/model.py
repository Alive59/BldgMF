import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal embeddings for time/position."""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] or [B, 1] time values in [0, 1]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -np.log(self.max_period) * 
            torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        
        args = t * freqs.unsqueeze(0) * 1000  # Scale for better gradients
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on time embedding.
    
    Common Mistake #4: Using standard LayerNorm without time conditioning.
    Solution: Use AdaLN for proper time-dependent normalization.
    """
    
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # Predict scale and shift from condition
        self.ada_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_dim * 2),
        )
        
        # Initialize to identity transform
        nn.init.zeros_(self.ada_linear[-1].weight)
        nn.init.zeros_(self.ada_linear[-1].bias)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input features
            cond: [B, D_cond] condition embedding
        """
        # Get adaptive parameters
        params = self.ada_linear(cond)  # [B, 2*D]
        scale, shift = params.chunk(2, dim=-1)  # [B, D] each
        
        # Normalize and apply adaptive transform
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x


class MeshAttentionBlock(nn.Module):
    """
    Attention block with mesh structure awareness.
    
    Common Mistake #5: Treating mesh vertices as independent points.
    Solution: Incorporate edge/face connectivity in attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        condition_dim: int,
        use_edge_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_edge_bias = use_edge_bias
        
        # Adaptive layer norms
        self.norm1 = AdaptiveLayerNorm(hidden_dim, condition_dim)
        self.norm2 = AdaptiveLayerNorm(hidden_dim, condition_dim)
        
        # Self-attention
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge bias (learned attention bias based on mesh connectivity)
        if use_edge_bias:
            self.edge_bias_embed = nn.Embedding(4, num_heads)  # 0: no edge, 1: 1-ring, 2: 2-ring, 3: same vertex
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        edge_index: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, V, D] vertex features
            cond: [B, D_cond] condition (time + other)
            edge_index: [B, V, V] mesh adjacency (0/1/2/3 for distance)
            attention_mask: [B, V] valid vertex mask
        """
        B, V, D = x.shape
        
        # Self-attention with AdaLN
        h = self.norm1(x, cond)
        
        qkv = self.qkv(h).reshape(B, V, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, V, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, heads, V, V]
        
        # Add edge bias (mesh-aware attention)
        if self.use_edge_bias and edge_index is not None:
            edge_bias = self.edge_bias_embed(edge_index)  # [B, V, V, heads]
            edge_bias = edge_bias.permute(0, 3, 1, 2)  # [B, heads, V, V]
            attn = attn + edge_bias
        
        # Mask invalid vertices
        if attention_mask is not None:
            # attention_mask: [B, V] -> [B, 1, 1, V]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # [B, heads, V, head_dim]
        out = out.transpose(1, 2).reshape(B, V, D)
        out = self.proj(out)
        
        x = x + out
        
        # MLP with AdaLN
        x = x + self.mlp(self.norm2(x, cond))
        
        return x


class MeshCrossAttentionBlock(nn.Module):
    """Cross-attention for conditioning (e.g., from footprint or image)."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        condition_dim: int,
        context_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.norm1 = AdaptiveLayerNorm(hidden_dim, condition_dim)
        self.norm_context = nn.LayerNorm(context_dim)
        
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv = nn.Linear(context_dim, hidden_dim * 2, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        cond: torch.Tensor,
        context_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, V, D] vertex features
            context: [B, L, D_ctx] context features (e.g., footprint points)
            cond: [B, D_cond] time condition
            context_mask: [B, L] valid context mask
        """
        B, V, D = x.shape
        L = context.shape[1]
        
        h = self.norm1(x, cond)
        context = self.norm_context(context)
        
        q = self.q(h).reshape(B, V, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(B, L, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if context_mask is not None:
            mask = context_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, V, D)
        out = self.proj(out)
        
        return x + out
    

class MeshMeanFlowNet(nn.Module):
    """
    DEPRECATED: Use FaceCentricMeshMeanFlowNet instead.

    Complete Mesh MeanFlow network for x-prediction (vertex-indexed version).

    Key design: Network outputs denoised vertex positions directly.
    """

    def __init__(
        self,
        max_vertices: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        condition_type: str = 'footprint',  # 'footprint', 'image', 'both'
        footprint_dim: int = 64,
        image_encoder: str = 'clip',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_vertices = max_vertices
        self.hidden_dim = hidden_dim
        self.condition_type = condition_type

        # ========== Input Embedding ==========
        # Vertex position embedding
        self.vertex_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable vertex position encoding (like ViT's position embedding)
        # Common Mistake #6: Not using position encoding for vertices
        self.vertex_pos_embed = nn.Parameter(torch.randn(1, max_vertices, hidden_dim) * 0.02)

        # ========== Time Embedding ==========
        # Common Mistake #7: Not properly encoding the (r, t) interval
        # Solution: Encode both t, r, and (t-r) for the MeanFlow formulation
        time_embed_dim = hidden_dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Interval embedding (t - r)
        self.interval_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Combined condition projection
        self.cond_proj = nn.Linear(time_embed_dim * 2, hidden_dim)

        # ========== Condition Encoders ==========
        if condition_type in ['footprint', 'both']:
            self.footprint_encoder = FootprintEncoder(
                input_dim=2,
                hidden_dim=footprint_dim,
                output_dim=hidden_dim,
            )

        if condition_type in ['image', 'both']:
            self.image_encoder = ImageConditionEncoder(
                encoder_type=image_encoder,
                output_dim=hidden_dim,
            )

        # ========== Transformer Blocks ==========
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Self-attention
            self.blocks.append(
                MeshAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    condition_dim=hidden_dim,
                    use_edge_bias=True,
                    dropout=dropout,
                )
            )

            # Cross-attention every 2 layers
            if i % 2 == 1 and condition_type != 'none':
                self.blocks.append(
                    MeshCrossAttentionBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        condition_dim=hidden_dim,
                        context_dim=hidden_dim,
                        dropout=dropout,
                    )
                )

        # ========== Output Head ==========
        # Common Mistake #8: Using single linear layer for output
        # Solution: Use a proper output head with residual connection to input
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Initialize output to zero (start with identity mapping)
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

        # ========== Mesh Adjacency Encoder ==========
        self.adjacency_cache = {}

    def compute_edge_index(self, faces: torch.Tensor, num_vertices: int) -> torch.Tensor:
        """
        Compute vertex adjacency matrix from faces.

        Returns:
            edge_index: [V, V] with values 0 (no edge), 1 (1-ring neighbor),
                       2 (2-ring neighbor), 3 (same vertex)
        """
        # Cache for efficiency
        cache_key = (faces.shape, num_vertices)
        if cache_key in self.adjacency_cache:
            return self.adjacency_cache[cache_key].to(faces.device)

        V = num_vertices
        adj = torch.zeros(V, V, dtype=torch.long, device=faces.device)

        # Diagonal (same vertex)
        adj.fill_diagonal_(3)

        # 1-ring neighbors (share an edge)
        for i in range(3):
            for j in range(3):
                if i != j:
                    idx_i = faces[:, i]
                    idx_j = faces[:, j]
                    adj[idx_i, idx_j] = 1

        # 2-ring neighbors (share a face but not an edge)
        for i in range(3):
            for j in range(3):
                if i != j:
                    idx_i = faces[:, i]
                    idx_j = faces[:, j]
                    # Where not already 1-ring or same
                    mask = adj[idx_i, idx_j] == 0
                    if mask.any():
                        adj[idx_i[mask], idx_j[mask]] = 2

        self.adjacency_cache[cache_key] = adj.cpu()
        return adj

    def forward(
        self,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        faces: torch.Tensor,
        vertex_mask: torch.Tensor = None,
        footprint: torch.Tensor = None,
        footprint_mask: torch.Tensor = None,
        image: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict denoised vertices (x-prediction).

        Args:
            z_t: [B, V, 3] noisy vertex positions
            r: [B] or [B, 1] start time
            t: [B] or [B, 1] end time
            faces: [B, F, 3] or [F, 3] face indices
            vertex_mask: [B, V] valid vertex mask
            footprint: [B, P, 2] footprint polygon points
            footprint_mask: [B, P] valid footprint point mask
            image: [B, 3, H, W] conditioning image

        Returns:
            x_pred: [B, V, 3] predicted denoised vertices
        """
        B, V, _ = z_t.shape

        # Ensure time tensors have correct shape
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Default mask
        if vertex_mask is None:
            vertex_mask = torch.ones(B, V, dtype=torch.bool, device=z_t.device)

        # ========== Embed Time ==========
        t_emb = self.time_embed(t.squeeze(-1))  # [B, time_dim]
        r_emb = self.time_embed(r.squeeze(-1))
        interval_emb = self.interval_embed((t - r).squeeze(-1))

        # Combine time embeddings
        time_cond = torch.cat([t_emb + r_emb, interval_emb], dim=-1)
        cond = self.cond_proj(time_cond)  # [B, hidden_dim]

        # ========== Embed Vertices ==========
        h = self.vertex_embed(z_t)  # [B, V, hidden_dim]

        # Add position embedding
        h = h + self.vertex_pos_embed[:, :V, :]

        # ========== Encode Conditions ==========
        context = None
        context_mask = None

        if self.condition_type in ['footprint', 'both'] and footprint is not None:
            fp_feat = self.footprint_encoder(footprint, footprint_mask)  # [B, P, hidden]
            context = fp_feat
            context_mask = footprint_mask

        if self.condition_type in ['image', 'both'] and image is not None:
            img_feat = self.image_encoder(image)  # [B, L, hidden]
            if context is not None:
                context = torch.cat([context, img_feat], dim=1)
                img_mask = torch.ones(B, img_feat.shape[1], dtype=torch.bool, device=image.device)
                context_mask = torch.cat([context_mask, img_mask], dim=1)
            else:
                context = img_feat
                context_mask = torch.ones(B, img_feat.shape[1], dtype=torch.bool, device=image.device)

        # ========== Compute Edge Index ==========
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(B, -1, -1)

        # Use first batch's faces for adjacency (assuming same topology)
        edge_index = self.compute_edge_index(faces[0], V)
        edge_index = edge_index.unsqueeze(0).expand(B, -1, -1)

        # ========== Transformer Blocks ==========
        for block in self.blocks:
            if isinstance(block, MeshAttentionBlock):
                h = block(h, cond, edge_index, vertex_mask)
            elif isinstance(block, MeshCrossAttentionBlock):
                if context is not None:
                    h = block(h, context, cond, context_mask)

        # ========== Output Head ==========
        h = self.output_norm(h)
        delta = self.output_head(h)  # [B, V, 3]

        # Common Mistake #9: Directly outputting vertices without residual
        # Solution: Predict residual/delta from noisy input for better training
        # This is similar to "v-prediction" vs "x-prediction" nuance
        # Here we predict the "clean" x directly, but initialize to pass-through
        x_pred = z_t + delta

        # Mask invalid vertices
        x_pred = x_pred * vertex_mask.unsqueeze(-1)

        return x_pred


class FaceCentricMeshMeanFlowNet(nn.Module):
    """
    Face-centric Mesh MeanFlow network for x-prediction.

    Key design:
    - Input/output shape: [B, F, 3, 3] (F faces, 3 vertices per face, 3 coords)
    - Internally reshaped to [B, F*3, 3] for transformer processing
    - No need for separate face indices - topology is implicit
    """

    def __init__(
        self,
        max_faces: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        condition_type: str = 'footprint',  # 'footprint', 'image', 'both', 'none'
        footprint_dim: int = 64,
        image_encoder: str = 'clip',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_faces = max_faces
        self.hidden_dim = hidden_dim
        self.condition_type = condition_type

        # ========== Input Embedding ==========
        # Vertex position embedding (same as before)
        self.vertex_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Face-aware position encoding
        # Encodes (face_index, vertex_in_face) information
        self.face_pos_embed = nn.Parameter(
            torch.randn(1, max_faces, hidden_dim) * 0.02
        )
        # Vertex-in-face embedding (0, 1, 2 for each vertex in a face)
        self.vertex_in_face_embed = nn.Parameter(
            torch.randn(1, 3, hidden_dim) * 0.02
        )

        # ========== Time Embedding ==========
        time_embed_dim = hidden_dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Interval embedding (t - r)
        self.interval_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Combined condition projection
        self.cond_proj = nn.Linear(time_embed_dim * 2, hidden_dim)

        # ========== Condition Encoders ==========
        if condition_type in ['footprint', 'both']:
            self.footprint_encoder = FootprintEncoder(
                input_dim=2,
                hidden_dim=footprint_dim,
                output_dim=hidden_dim,
            )

        if condition_type in ['image', 'both']:
            self.image_encoder = ImageConditionEncoder(
                encoder_type=image_encoder,
                output_dim=hidden_dim,
            )

        # ========== Transformer Blocks ==========
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Self-attention (no edge bias for face-centric)
            self.blocks.append(
                MeshAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    condition_dim=hidden_dim,
                    use_edge_bias=False,  # No edge bias for face-centric
                    dropout=dropout,
                )
            )

            # Cross-attention every 2 layers
            if i % 2 == 1 and condition_type != 'none':
                self.blocks.append(
                    MeshCrossAttentionBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        condition_dim=hidden_dim,
                        context_dim=hidden_dim,
                        dropout=dropout,
                    )
                )

        # ========== Output Head ==========
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Initialize output to zero (start with identity mapping)
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(
        self,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        face_mask: torch.Tensor = None,
        footprint: torch.Tensor = None,
        footprint_mask: torch.Tensor = None,
        image: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict denoised face vertices (x-prediction).

        Args:
            z_t: [B, F, 3, 3] noisy face vertices (F faces, 3 vertices, 3 coords)
            r: [B] or [B, 1] start time
            t: [B] or [B, 1] end time
            face_mask: [B, F] valid face mask
            footprint: [B, P, 2] footprint polygon points
            footprint_mask: [B, P] valid footprint point mask
            image: [B, 3, H, W] conditioning image

        Returns:
            x_pred: [B, F, 3, 3] predicted denoised face vertices
        """
        B, F, _, _ = z_t.shape

        # Ensure time tensors have correct shape
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Default mask
        if face_mask is None:
            face_mask = torch.ones(B, F, dtype=torch.bool, device=z_t.device)

        # ========== Reshape to [B, F*3, 3] ==========
        z_flat = z_t.reshape(B, F * 3, 3)  # [B, F*3, 3]

        # ========== Embed Time ==========
        t_emb = self.time_embed(t.squeeze(-1))  # [B, time_dim]
        r_emb = self.time_embed(r.squeeze(-1))
        interval_emb = self.interval_embed((t - r).squeeze(-1))

        # Combine time embeddings
        time_cond = torch.cat([t_emb + r_emb, interval_emb], dim=-1)
        cond = self.cond_proj(time_cond)  # [B, hidden_dim]

        # ========== Embed Vertices ==========
        h = self.vertex_embed(z_flat)  # [B, F*3, hidden_dim]

        # Add face position embedding (broadcast to 3 vertices per face)
        face_pos = self.face_pos_embed[:, :F, :]  # [1, F, hidden]
        face_pos = face_pos.unsqueeze(2).expand(-1, -1, 3, -1)  # [1, F, 3, hidden]
        face_pos = face_pos.reshape(1, F * 3, -1)  # [1, F*3, hidden]
        h = h + face_pos

        # Add vertex-in-face embedding: [v0, v1, v2, v0, v1, v2, ...] repeated F times
        vif_pos = self.vertex_in_face_embed.expand(F, -1, -1).reshape(1, F * 3, -1)  # [1, F*3, hidden]
        h = h + vif_pos

        # ========== Create attention mask ==========
        # Expand face_mask to cover all 3 vertices per face
        attention_mask = face_mask.unsqueeze(-1).expand(-1, -1, 3)  # [B, F, 3]
        attention_mask = attention_mask.reshape(B, F * 3)  # [B, F*3]

        # ========== Encode Conditions ==========
        context = None
        context_mask = None
        
        if self.condition_type in ['footprint', 'both'] and footprint is not None:
            fp_feat = self.footprint_encoder(footprint, footprint_mask)  # [B, P, hidden]
            context = fp_feat
            context_mask = footprint_mask

        if self.condition_type in ['image', 'both'] and image is not None:
            img_feat = self.image_encoder(image)  # [B, L, hidden]
            if context is not None:
                context = torch.cat([context, img_feat], dim=1)
                img_mask = torch.ones(B, img_feat.shape[1], dtype=torch.bool, device=image.device)
                context_mask = torch.cat([context_mask, img_mask], dim=1)
            else:
                context = img_feat
                context_mask = torch.ones(B, img_feat.shape[1], dtype=torch.bool, device=image.device)

        # ========== Transformer Blocks ==========
        for block in self.blocks:
            if isinstance(block, MeshAttentionBlock):
                h = block(h, cond, edge_index=None, attention_mask=attention_mask)
            elif isinstance(block, MeshCrossAttentionBlock):
                if context is not None:
                    h = block(h, context, cond, context_mask)

        # ========== Output Head ==========
        h = self.output_norm(h)
        delta = self.output_head(h)  # [B, F*3, 3]

        # Add residual
        x_pred_flat = z_flat + delta

        # ========== Reshape back to [B, F, 3, 3] ==========
        x_pred = x_pred_flat.reshape(B, F, 3, 3)

        # Mask invalid faces
        x_pred = x_pred * face_mask.unsqueeze(-1).unsqueeze(-1)

        return x_pred


class FootprintEncoder(nn.Module):
    """Encode 2D building footprint polygon."""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 512):
        super().__init__()
        
        self.point_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=8,
                dim_feedforward=output_dim * 4,
                batch_first=True,
            ),
            num_layers=4,
        )
        
    def forward(self, footprint: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            footprint: [B, P, 2] polygon points
            mask: [B, P] valid point mask
        """
        h = self.point_embed(footprint)
        
        if mask is not None:
            # Create attention mask (True = ignore)
            attn_mask = ~mask
            h = self.transformer(h, src_key_padding_mask=attn_mask)
        else:
            h = self.transformer(h)
        
        return h


class ImageConditionEncoder(nn.Module):
    """Encode conditioning image (satellite, etc.)."""

    def __init__(self, encoder_type: str = 'dinov2', output_dim: int = 512):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == 'dinov2':
            self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            encoder_dim = 768
        elif encoder_type == 'clip':
            import clip
            self.encoder, _ = clip.load('ViT-B/32', device='cpu')
            self.encoder = self.encoder.float()
            # CLIP ViT-B/32: 768 internal dim, projected to 512 via visual.proj
            # We use the 512-d projected patch tokens
            encoder_dim = 512
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.proj = nn.Linear(encoder_dim, output_dim)

    def _clip_patch_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract [B, num_patches, 512] from CLIP visual encoder."""
        v = self.encoder.visual
        x = v.conv1(image)                          # [B, 768, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)   # [B, 768, grid**2]
        x = x.permute(0, 2, 1)                       # [B, grid**2, 768]
        cls = v.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)               # [B, grid**2+1, 768]
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)                       # LND
        x = v.transformer(x)
        x = x.permute(1, 0, 2)                       # NLD
        # Apply ln_post and projection to ALL tokens, then drop CLS
        x = v.ln_post(x)
        if v.proj is not None:
            x = x @ v.proj                            # [B, grid**2+1, 512]
        return x[:, 1:, :]                            # [B, grid**2, 512]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W]

        Returns:
            features: [B, L, output_dim]
        """
        with torch.no_grad():
            if self.encoder_type == 'clip':
                features = self._clip_patch_features(image)
            else:
                # DINOv2 path
                features = self.encoder.forward_features(image)
                if isinstance(features, dict):
                    features = features['x_norm_patchtokens']

        features = self.proj(features)
        return features