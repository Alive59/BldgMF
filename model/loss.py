import torch
import torch.nn as nn
import torch.nn.functional as F

from data.data import compute_face_normals


class MeshGeometricLoss(nn.Module):
    """
    Geometric regularization for mesh quality.
    
    Common Mistake #16: Not regularizing mesh quality, leading to 
    self-intersections, degenerate faces, non-manifold geometry.
    """
    
    def __init__(
        self,
        lambda_face_area: float = 0.1,
        lambda_aspect_ratio: float = 0.1,
        lambda_dihedral: float = 0.05,
        min_face_area: float = 1e-6,
        max_aspect_ratio: float = 10.0,
    ):
        super().__init__()
        self.lambda_face_area = lambda_face_area
        self.lambda_aspect_ratio = lambda_aspect_ratio
        self.lambda_dihedral = lambda_dihedral
        self.min_face_area = min_face_area
        self.max_aspect_ratio = max_aspect_ratio
    
    def forward(
        self,
        pred_vertices: torch.Tensor,
        gt_vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute geometric regularization losses."""
        
        total_loss = 0.0
        
        # Face area regularization (avoid degenerate faces)
        if self.lambda_face_area > 0:
            area_loss = self.face_area_loss(pred_vertices, faces)
            total_loss = total_loss + self.lambda_face_area * area_loss
        
        # Aspect ratio regularization (avoid thin triangles)
        if self.lambda_aspect_ratio > 0:
            aspect_loss = self.aspect_ratio_loss(pred_vertices, faces)
            total_loss = total_loss + self.lambda_aspect_ratio * aspect_loss
        
        # Dihedral angle regularization (smoothness)
        if self.lambda_dihedral > 0:
            dihedral_loss = self.dihedral_angle_loss(pred_vertices, faces)
            total_loss = total_loss + self.lambda_dihedral * dihedral_loss
        
        return total_loss
    
    def face_area_loss(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize faces with area below threshold."""
        areas = self.compute_face_areas(vertices, faces)
        
        # Soft penalty for small faces
        loss = F.relu(self.min_face_area - areas).mean()
        
        return loss
    
    def compute_face_areas(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """Compute area of each face."""
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(vertices.shape[0], -1, -1)
        
        v0 = torch.gather(vertices, 1, faces[:, :, 0:1].expand(-1, -1, 3))
        v1 = torch.gather(vertices, 1, faces[:, :, 1:2].expand(-1, -1, 3))
        v2 = torch.gather(vertices, 1, faces[:, :, 2:3].expand(-1, -1, 3))
        
        e1 = v1 - v0
        e2 = v2 - v0
        
        cross = torch.cross(e1, e2, dim=-1)
        areas = 0.5 * cross.norm(dim=-1)
        
        return areas
    
    def aspect_ratio_loss(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize triangles with high aspect ratio."""
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(vertices.shape[0], -1, -1)
        
        v0 = torch.gather(vertices, 1, faces[:, :, 0:1].expand(-1, -1, 3))
        v1 = torch.gather(vertices, 1, faces[:, :, 1:2].expand(-1, -1, 3))
        v2 = torch.gather(vertices, 1, faces[:, :, 2:3].expand(-1, -1, 3))
        
        # Edge lengths
        e1 = (v1 - v0).norm(dim=-1)
        e2 = (v2 - v1).norm(dim=-1)
        e3 = (v0 - v2).norm(dim=-1)
        
        # Aspect ratio = longest edge / shortest edge
        max_edge = torch.stack([e1, e2, e3], dim=-1).max(dim=-1)[0]
        min_edge = torch.stack([e1, e2, e3], dim=-1).min(dim=-1)[0].clamp(min=1e-8)
        
        aspect_ratio = max_edge / min_edge
        
        # Penalize high aspect ratios
        loss = F.relu(aspect_ratio - self.max_aspect_ratio).mean()
        
        return loss
    
    def dihedral_angle_loss(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Regularize dihedral angles between adjacent faces.
        Encourages smooth surfaces.
        """
        face_normals = compute_face_normals(vertices, faces)
        
        # Build face adjacency (faces sharing an edge)
        # This is expensive, so we use a simplified version
        # that penalizes variance in normals of nearby faces
        
        # Compute mean normal direction
        mean_normal = face_normals.mean(dim=1, keepdim=True)
        
        # Penalize deviation from mean (encourages smoothness)
        deviation = (face_normals - mean_normal).norm(dim=-1)
        loss = deviation.mean()
        
        return loss


class MeshPerceptualLoss(nn.Module):
    """
    Combined perceptual loss for meshes.
    
    Common Mistake #15: Only using Chamfer distance.
    Solution: Combine multiple geometric and perceptual metrics.
    """
    
    def __init__(
        self,
        use_chamfer: bool = True,
        use_normal: bool = True,
        use_edge: bool = True,
        use_laplacian: bool = True,
        use_render: bool = False,
        lambda_chamfer: float = 1.0,
        lambda_normal: float = 0.5,
        lambda_edge: float = 0.2,
        lambda_laplacian: float = 0.1,
        lambda_render: float = 0.5,
    ):
        super().__init__()
        
        self.use_chamfer = use_chamfer
        self.use_normal = use_normal
        self.use_edge = use_edge
        self.use_laplacian = use_laplacian
        self.use_render = use_render
        
        self.lambda_chamfer = lambda_chamfer
        self.lambda_normal = lambda_normal
        self.lambda_edge = lambda_edge
        self.lambda_laplacian = lambda_laplacian
        self.lambda_render = lambda_render
        
        if use_render:
            self.render_loss = RenderBasedPerceptualLoss()
    
    def forward(
        self,
        pred_vertices: torch.Tensor,
        gt_vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_vertices: [B, V, 3]
            gt_vertices: [B, V, 3]
            faces: [B, F, 3]
            vertex_mask: [B, V]
        """
        total_loss = 0.0
        
        if self.use_chamfer:
            chamfer = self.chamfer_loss(pred_vertices, gt_vertices, vertex_mask)
            total_loss = total_loss + self.lambda_chamfer * chamfer
        
        if self.use_normal:
            normal = self.normal_consistency_loss(pred_vertices, gt_vertices, faces)
            total_loss = total_loss + self.lambda_normal * normal
        
        if self.use_edge:
            edge = self.edge_length_loss(pred_vertices, gt_vertices, faces)
            total_loss = total_loss + self.lambda_edge * edge
        
        if self.use_laplacian:
            laplacian = self.laplacian_smoothing_loss(pred_vertices, gt_vertices, faces)
            total_loss = total_loss + self.lambda_laplacian * laplacian
        
        if self.use_render:
            render = self.render_loss(pred_vertices, gt_vertices, faces)
            total_loss = total_loss + self.lambda_render * render
        
        return total_loss
    
    def chamfer_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Bidirectional Chamfer distance."""
        if mask is not None:
            # Only consider valid vertices
            pred = pred * mask.unsqueeze(-1)
            gt = gt * mask.unsqueeze(-1)
        
        # pred -> gt
        dist_p2g = torch.cdist(pred, gt)  # [B, V, V]
        min_p2g = dist_p2g.min(dim=-1)[0]  # [B, V]
        
        # gt -> pred
        min_g2p = dist_p2g.min(dim=-2)[0]  # [B, V]
        
        if mask is not None:
            min_p2g = min_p2g * mask
            min_g2p = min_g2p * mask
            loss = (min_p2g.sum() + min_g2p.sum()) / (2 * mask.sum())
        else:
            loss = (min_p2g.mean() + min_g2p.mean()) / 2
        
        return loss
    
    def normal_consistency_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize difference in face normals."""
        pred_normals = compute_face_normals(pred, faces)
        gt_normals = compute_face_normals(gt, faces)
        
        # Cosine similarity loss
        cos_sim = (pred_normals * gt_normals).sum(dim=-1)
        loss = (1 - cos_sim).mean()
        
        return loss
    
    def edge_length_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize difference in edge lengths."""
        def get_edge_lengths(vertices, faces):
            if faces.dim() == 2:
                faces = faces.unsqueeze(0).expand(vertices.shape[0], -1, -1)
            
            v0 = torch.gather(vertices, 1, faces[:, :, 0:1].expand(-1, -1, 3))
            v1 = torch.gather(vertices, 1, faces[:, :, 1:2].expand(-1, -1, 3))
            v2 = torch.gather(vertices, 1, faces[:, :, 2:3].expand(-1, -1, 3))
            
            e1 = (v1 - v0).norm(dim=-1)
            e2 = (v2 - v1).norm(dim=-1)
            e3 = (v0 - v2).norm(dim=-1)
            
            return torch.stack([e1, e2, e3], dim=-1)
        
        pred_edges = get_edge_lengths(pred, faces)
        gt_edges = get_edge_lengths(gt, faces)
        
        loss = F.mse_loss(pred_edges, gt_edges)
        return loss
    
    def laplacian_smoothing_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Laplacian smoothing regularization.
        Encourages similar local geometry.
        """
        pred_lap = self.compute_laplacian(pred, faces)
        gt_lap = self.compute_laplacian(gt, faces)
        
        loss = F.mse_loss(pred_lap, gt_lap)
        return loss
    
    def compute_laplacian(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Laplacian coordinates (difference from neighbor centroid)."""
        B, V, _ = vertices.shape
        
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(B, -1, -1)
        
        # Build adjacency and compute neighbor sum
        neighbor_sum = torch.zeros_like(vertices)
        neighbor_count = torch.zeros(B, V, 1, device=vertices.device)
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    src_idx = faces[:, :, i]
                    dst_idx = faces[:, :, j]
                    
                    # Gather source vertices
                    src_verts = torch.gather(
                        vertices, 1, 
                        src_idx.unsqueeze(-1).expand(-1, -1, 3)
                    )
                    
                    # Scatter add to destinations
                    neighbor_sum.scatter_add_(
                        1,
                        dst_idx.unsqueeze(-1).expand(-1, -1, 3),
                        src_verts
                    )
                    neighbor_count.scatter_add_(
                        1,
                        dst_idx.unsqueeze(-1),
                        torch.ones_like(dst_idx.unsqueeze(-1).float())
                    )
        
        neighbor_count = neighbor_count.clamp(min=1)
        neighbor_centroid = neighbor_sum / neighbor_count
        
        laplacian = vertices - neighbor_centroid
        return laplacian


class RenderBasedPerceptualLoss(nn.Module):
    """
    Render mesh to images and apply 2D perceptual loss.
    
    This bridges 3D mesh quality to 2D perceptual metrics.
    """
    
    def __init__(
        self,
        num_views: int = 8,
        image_size: int = 256,
        use_lpips: bool = True,
    ):
        super().__init__()
        self.num_views = num_views
        self.image_size = image_size
        
        if use_lpips:
            import lpips
            self.lpips = lpips.LPIPS(net='vgg')
            for param in self.lpips.parameters():
                param.requires_grad = False
        else:
            self.lpips = None
        
        # Setup differentiable renderer (using PyTorch3D)
        self._setup_renderer()
    
    def _setup_renderer(self):
        """Setup PyTorch3D renderer."""
        try:
            from pytorch3d.renderer import (
                look_at_view_transform,
                FoVPerspectiveCameras,
                RasterizationSettings,
                MeshRenderer,
                MeshRasterizer,
                SoftPhongShader,
                PointLights,
            )
            
            self.raster_settings = RasterizationSettings(
                image_size=self.image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            
            # Pre-compute camera positions (evenly distributed on sphere)
            elevations = [30, 30, 30, 30, -30, -30, -30, -30][:self.num_views]
            azimuths = [0, 90, 180, 270, 45, 135, 225, 315][:self.num_views]
            
            self.camera_positions = list(zip(elevations, azimuths))
            self._pytorch3d_available = True
            
        except ImportError:
            print("PyTorch3D not available, render loss disabled")
            self._pytorch3d_available = False
    
    def forward(
        self,
        pred_vertices: torch.Tensor,
        gt_vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """Render and compare."""
        if not self._pytorch3d_available:
            return torch.tensor(0.0, device=pred_vertices.device)
        
        total_loss = 0.0
        
        for elev, azim in self.camera_positions:
            pred_img = self._render(pred_vertices, faces, elev, azim)
            gt_img = self._render(gt_vertices, faces, elev, azim)
            
            if self.lpips is not None:
                # LPIPS expects [B, 3, H, W] in [-1, 1]
                loss = self.lpips(pred_img * 2 - 1, gt_img * 2 - 1).mean()
            else:
                loss = F.mse_loss(pred_img, gt_img)
            
            total_loss = total_loss + loss
        
        return total_loss / self.num_views
    
    def _render(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        elevation: float,
        azimuth: float,
    ) -> torch.Tensor:
        """Render mesh from given viewpoint."""
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import (
            look_at_view_transform,
            FoVPerspectiveCameras,
            MeshRenderer,
            MeshRasterizer,
            SoftPhongShader,
            PointLights,
            TexturesVertex,
        )
        
        device = vertices.device
        B = vertices.shape[0]
        
        # Create mesh
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(B, -1, -1)
        
        # Simple vertex colors (white)
        verts_rgb = torch.ones_like(vertices)
        textures = TexturesVertex(verts_features=verts_rgb)
        
        meshes = Meshes(verts=vertices, faces=faces, textures=textures)
        
        # Camera
        R, T = look_at_view_transform(dist=2.5, elev=elevation, azim=azimuth)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        
        # Lights
        lights = PointLights(device=device, location=[[0, 2, 2]])
        
        # Renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings,
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
            )
        )
        
        images = renderer(meshes)  # [B, H, W, 4]
        images = images[..., :3].permute(0, 3, 1, 2)  # [B, 3, H, W]
        
        return images