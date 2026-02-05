import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass
class MeshData:
    """
    Standardized mesh representation.
    
    Common Mistake #1: Using inconsistent mesh formats throughout the pipeline.
    Solution: Define a clear data structure used everywhere.
    """
    vertices: torch.Tensor      # [B, V, 3] vertex positions
    faces: torch.Tensor         # [B, F, 3] or [F, 3] face indices (int64)
    vertex_mask: torch.Tensor   # [B, V] valid vertex mask (for variable size)
    face_mask: torch.Tensor     # [B, F] valid face mask
    
    # Optional attributes
    vertex_normals: Optional[torch.Tensor] = None   # [B, V, 3]
    face_normals: Optional[torch.Tensor] = None     # [B, F, 3]
    vertex_features: Optional[torch.Tensor] = None  # [B, V, D]
    
    def __post_init__(self):
        """Validate mesh data."""
        B, V, _ = self.vertices.shape
        
        # Ensure faces reference valid vertices
        if self.faces.max() >= V:
            raise ValueError(f"Face index {self.faces.max()} >= num vertices {V}")
        
        # Ensure correct dtypes
        assert self.faces.dtype == torch.int64, "Faces must be int64"
        assert self.vertices.dtype == torch.float32, "Vertices must be float32"


class MeshNormalizer:
    """
    Normalize meshes to canonical space.
    
    Common Mistake #2: Not normalizing coordinates, leading to scale/position variance.
    Solution: Always normalize to [-1, 1] or unit sphere during training.
    """
    
    def __init__(self, method: str = 'unit_cube'):
        """
        Args:
            method: 'unit_cube' | 'unit_sphere' | 'per_axis'
        """
        self.method = method
        
    def normalize(self, vertices: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Normalize vertices to canonical space.
        
        Args:
            vertices: [B, V, 3]
            mask: [B, V] valid vertex mask
            
        Returns:
            normalized_vertices: [B, V, 3]
            stats: dict with normalization parameters for denormalization
        """
        B, V, _ = vertices.shape
        
        if mask is None:
            mask = torch.ones(B, V, dtype=torch.bool, device=vertices.device)
        
        # Compute per-batch statistics only on valid vertices
        stats = {}
        
        if self.method == 'unit_cube':
            # Center and scale to [-1, 1]
            masked_verts = vertices.clone()
            masked_verts[~mask] = float('nan')
            
            # Compute min/max ignoring invalid vertices
            v_min = torch.nanmin(masked_verts, dim=1, keepdim=True).values  # [B, 1, 3]
            v_max = torch.nanmax(masked_verts, dim=1, keepdim=True).values
            
            center = (v_min + v_max) / 2
            scale = (v_max - v_min).max(dim=-1, keepdim=True).values / 2
            scale = scale.clamp(min=1e-6)  # Avoid division by zero
            
            normalized = (vertices - center) / scale
            
            stats = {'center': center, 'scale': scale}
            
        elif self.method == 'unit_sphere':
            # Center and scale to unit sphere
            masked_verts = vertices.clone()
            masked_verts[~mask] = 0
            
            # Centroid
            valid_count = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            center = (masked_verts * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_count
            
            centered = vertices - center
            
            # Max radius
            distances = (centered ** 2).sum(dim=-1, keepdim=True).sqrt()
            distances[~mask] = 0
            scale = distances.max(dim=1, keepdim=True).values.clamp(min=1e-6)
            
            normalized = centered / scale
            
            stats = {'center': center, 'scale': scale}
            
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        # Zero out invalid vertices
        normalized[~mask] = 0
        
        return normalized, stats
    
    def denormalize(self, vertices: torch.Tensor, stats: Dict) -> torch.Tensor:
        """Reverse normalization."""
        return vertices * stats['scale'] + stats['center']


def compute_face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute face normals from vertices and faces.
    
    Common Mistake #3: Inconsistent winding order leading to flipped normals.
    Solution: Use consistent counter-clockwise winding, validate during data loading.
    
    Args:
        vertices: [B, V, 3]
        faces: [B, F, 3] or [F, 3]
        
    Returns:
        face_normals: [B, F, 3] normalized face normals
    """
    if faces.dim() == 2:
        faces = faces.unsqueeze(0).expand(vertices.shape[0], -1, -1)
    
    # Gather vertices for each face
    # faces: [B, F, 3] indices
    v0 = torch.gather(vertices, 1, faces[:, :, 0:1].expand(-1, -1, 3))  # [B, F, 3]
    v1 = torch.gather(vertices, 1, faces[:, :, 1:2].expand(-1, -1, 3))
    v2 = torch.gather(vertices, 1, faces[:, :, 2:3].expand(-1, -1, 3))
    
    # Edge vectors
    e1 = v1 - v0
    e2 = v2 - v0
    
    # Cross product (counter-clockwise winding)
    normals = torch.cross(e1, e2, dim=-1)
    
    # Normalize (handle degenerate faces)
    norm = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normals = normals / norm
    
    return normals


def compute_vertex_normals(
    vertices: torch.Tensor, 
    faces: torch.Tensor,
    face_normals: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute vertex normals by averaging adjacent face normals.
    
    Args:
        vertices: [B, V, 3]
        faces: [B, F, 3] or [F, 3]
        face_normals: [B, F, 3] optional precomputed face normals
        
    Returns:
        vertex_normals: [B, V, 3]
    """
    B, V, _ = vertices.shape
    
    if faces.dim() == 2:
        faces = faces.unsqueeze(0).expand(B, -1, -1)
    
    if face_normals is None:
        face_normals = compute_face_normals(vertices, faces)
    
    F = faces.shape[1]
    
    # Accumulate face normals to vertices
    vertex_normals = torch.zeros_like(vertices)
    
    # Scatter add face normals to each vertex
    for i in range(3):
        idx = faces[:, :, i].unsqueeze(-1).expand(-1, -1, 3)  # [B, F, 3]
        vertex_normals.scatter_add_(1, idx, face_normals)
    
    # Normalize
    norm = vertex_normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    vertex_normals = vertex_normals / norm
    
    return vertex_normals