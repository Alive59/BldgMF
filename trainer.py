import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp
from typing import Tuple, Dict

from model.model import MeshMeanFlowNet, FaceCentricMeshMeanFlowNet
from data.data import MeshNormalizer


class MeshMeanFlowTrainer:
    """
    Complete training logic for Mesh MeanFlow.
    
    Common Mistake #10: Incorrect JVP computation for MeanFlow.
    Solution: Carefully implement the derivative computation.
    """
    
    def __init__(
        self,
        model: MeshMeanFlowNet,
        optimizer: torch.optim.Optimizer,
        perceptual_loss: nn.Module = None,
        geometric_loss: nn.Module = None,
        lambda_perceptual: float = 0.4,
        lambda_geometric: float = 0.1,
        noise_schedule: str = 'logit_normal',
        r_ratio: float = 0.5,  # Ratio of r != t samples
    ):
        self.model = model
        self.optimizer = optimizer
        self.perceptual_loss = perceptual_loss
        self.geometric_loss = geometric_loss
        self.lambda_perceptual = lambda_perceptual
        self.lambda_geometric = lambda_geometric
        self.noise_schedule = noise_schedule
        self.r_ratio = r_ratio
        
        self.normalizer = MeshNormalizer(method='unit_cube')
        
    def sample_time(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (r, t) time pairs.
        
        Common Mistake #11: Always sampling r = t (degenerates to Flow Matching).
        Solution: Sample r from [0, t] for proper MeanFlow training.
        """
        if self.noise_schedule == 'logit_normal':
            # Logit-normal distribution for t
            u = torch.randn(batch_size, device=device)
            t = torch.sigmoid(0.8 * u + 0.8)
        elif self.noise_schedule == 'uniform':
            t = torch.rand(batch_size, device=device)
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
        
        t = t.clamp(1e-5, 1 - 1e-5)
        
        # Sample r
        # With probability (1 - r_ratio), r = t (Flow Matching mode)
        # With probability r_ratio, r ~ Uniform(0, t)
        use_interval = torch.rand(batch_size, device=device) < self.r_ratio
        r_uniform = t * torch.rand(batch_size, device=device)
        r = torch.where(use_interval, r_uniform, t)
        
        return r.unsqueeze(-1), t.unsqueeze(-1)
    
    def u_from_x(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Convert x-prediction to average velocity u."""
        # u = (z - x) / (t - r) (r always is 0)
        return (z - x_pred) / t.unsqueeze(-1)
    
    def compute_meanflow_loss(
        self,
        model: MeshMeanFlowNet,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        v_gt: torch.Tensor,
        faces: torch.Tensor,
        vertex_mask: torch.Tensor,
        condition_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MeanFlow loss with proper JVP.
        
        Common Mistake #12: Not using stop_gradient on du/dt term.
        Solution: Detach the JVP result as in the paper.
        """
        # Get x prediction
        x_pred = model(z_t, r, t, faces, vertex_mask, **condition_kwargs)
        
        # Convert to u
        u = self.u_from_x(z_t, t, x_pred)
        
        # Compute instantaneous velocity (r = t case)
        x_at_t = model(z_t, t, t, faces, vertex_mask, **condition_kwargs)
        v_pred = self.u_from_x(z_t, t, x_at_t)
        
        # Compute du/dt via finite difference (simpler than JVP for clarity)
        # Alternative: Use torch.func.jvp for exact derivative
        eps = 1e-4
        t_plus = t + eps
        x_plus = model(z_t, r, t_plus, faces, vertex_mask, **condition_kwargs)
        u_plus = self.u_from_x(z_t, t_plus, x_plus)
        
        dudt = (u_plus - u) / eps
        
        # MeanFlow compound function: V = u + (t - r) * du/dt
        # IMPORTANT: Stop gradient on du/dt
        V = u + (t - r).unsqueeze(-1) * dudt.detach()
        
        # MSE loss in velocity space
        loss = F.mse_loss(V, v_gt, reduction='none')
        
        # Mask invalid vertices
        loss = loss * vertex_mask.unsqueeze(-1)
        loss = loss.sum() / vertex_mask.sum() / 3
        
        return loss, x_pred
    
    def compute_meanflow_loss_with_jvp(
        self,
        model: MeshMeanFlowNet,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        v_gt: torch.Tensor,
        faces: torch.Tensor,
        vertex_mask: torch.Tensor,
        condition_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MeanFlow loss with proper JVP (more accurate than finite diff).
        
        Common Mistake #13: Incorrect tangent vector in JVP.
        Solution: Tangent for t should be 1, for others should be 0 or v.
        """
        # Define function for JVP
        def u_fn(z, r_val, t_val):
            x = model(z, r_val, t_val, faces, vertex_mask, **condition_kwargs)
            return (z - x) / t_val.unsqueeze(-1)
        
        # Compute instantaneous velocity at t (tangent direction)
        v_tangent = u_fn(z_t, t, t)
        
        # Compute u and du/dt via JVP
        # Tangents: (dz/dt = v, dr/dt = 0, dt/dt = 1)
        primals = (z_t, r, t)
        tangents = (v_tangent, torch.zeros_like(r), torch.ones_like(t))
        
        u, dudt = jvp(u_fn, primals, tangents)
        
        # MeanFlow: V = u + (t - r) * sg(du/dt)
        V = u + (t - r).unsqueeze(-1) * dudt.detach()
        
        # Loss
        loss = F.mse_loss(V, v_gt, reduction='none')
        loss = loss * vertex_mask.unsqueeze(-1)
        loss = loss.sum() / vertex_mask.sum() / 3
        
        # Also get x_pred for perceptual loss
        x_pred = z_t - t.unsqueeze(-1) * u
        
        return loss, x_pred
    
    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        Args:
            batch: {
                'vertices': [B, V, 3],
                'faces': [B, F, 3] or [F, 3],
                'vertex_mask': [B, V],
                'footprint': [B, P, 2] (optional),
                'image': [B, 3, H, W] (optional),
            }
        """
        self.model.train()
        
        vertices = batch['vertices']
        faces = batch['faces']
        vertex_mask = batch['vertex_mask']
        B, V, _ = vertices.shape
        device = vertices.device
        
        # Normalize vertices
        vertices_norm, norm_stats = self.normalizer.normalize(vertices, vertex_mask)
        
        # Sample time
        r, t = self.sample_time(B, device)
        
        # Sample noise and create noisy vertices
        epsilon = torch.randn_like(vertices_norm)
        z_t = (1 - t.unsqueeze(-1)) * vertices_norm + t.unsqueeze(-1) * epsilon  # linear interpolated target
        
        # Ground truth velocity
        v_gt = epsilon - vertices_norm
        
        # Prepare condition kwargs
        condition_kwargs = {}
        if 'footprint' in batch:
            condition_kwargs['footprint'] = batch['footprint']
            condition_kwargs['footprint_mask'] = batch.get('footprint_mask')
        if 'image' in batch:
            condition_kwargs['image'] = batch['image']
        
        # Compute MeanFlow loss
        velocity_loss, x_pred = self.compute_meanflow_loss(
            self.model, z_t, r, t, v_gt, faces, vertex_mask, condition_kwargs
        )
        
        total_loss = velocity_loss
        loss_dict = {'velocity_loss': velocity_loss.item()}
        
        # Perceptual loss (when noise is low)
        if self.perceptual_loss is not None:
            noise_mask = (t.squeeze(-1) < 0.8).float()
            if noise_mask.sum() > 0:
                # Denormalize for perceptual loss
                x_pred_denorm = self.normalizer.denormalize(x_pred, norm_stats)
                vertices_denorm = self.normalizer.denormalize(vertices_norm, norm_stats)
                
                perc_loss = self.perceptual_loss(
                    x_pred_denorm, vertices_denorm, faces, vertex_mask
                )
                perc_loss = perc_loss * noise_mask.mean()
                
                total_loss = total_loss + self.lambda_perceptual * perc_loss
                loss_dict['perceptual_loss'] = perc_loss.item()
        
        # Geometric loss
        if self.geometric_loss is not None:
            geo_loss = self.geometric_loss(x_pred, vertices_norm, faces, vertex_mask)
            total_loss = total_loss + self.lambda_geometric * geo_loss
            loss_dict['geometric_loss'] = geo_loss.item()
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        loss_dict['total_loss'] = total_loss.item()
        return loss_dict
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        faces: torch.Tensor,
        vertex_mask: torch.Tensor = None,
        condition_kwargs: Dict = None,
        device: torch.device = 'cuda',
    ) -> torch.Tensor:
        """
        One-step generation (1-NFE).
        
        Common Mistake #14: Using wrong time values for generation.
        Solution: t=1 (noise), r=0 (target clean), single step.
        """
        self.model.eval()
        
        V = self.model.max_vertices
        
        if vertex_mask is None:
            vertex_mask = torch.ones(batch_size, V, dtype=torch.bool, device=device)
        
        # Start from pure noise
        z_1 = torch.randn(batch_size, V, 3, device=device)
        z_1 = z_1 * vertex_mask.unsqueeze(-1)
        
        # One step: t=1, r=0
        t = torch.ones(batch_size, 1, device=device)
        r = torch.zeros(batch_size, 1, device=device)
        
        if condition_kwargs is None:
            condition_kwargs = {}
        
        # Predict denoised mesh
        x_pred = self.model(z_1, r, t, faces, vertex_mask, **condition_kwargs)

        return x_pred


class FaceCentricMeshMeanFlowTrainer:
    """
    Training logic for Face-Centric Mesh MeanFlow.

    Key differences from vertex-indexed version:
    - Input/output shape: [B, F, 3, 3] (F faces, 3 vertices per face, 3 coords)
    - No separate face indices needed - topology is implicit
    - Uses face_mask instead of vertex_mask
    """

    def __init__(
        self,
        model: FaceCentricMeshMeanFlowNet,
        optimizer: torch.optim.Optimizer,
        perceptual_loss: nn.Module = None,
        geometric_loss: nn.Module = None,
        lambda_perceptual: float = 0.4,
        lambda_geometric: float = 0.1,
        noise_schedule: str = 'logit_normal',
        r_ratio: float = 0.5,
    ):
        self.model = model
        self.optimizer = optimizer
        self.perceptual_loss = perceptual_loss
        self.geometric_loss = geometric_loss
        self.lambda_perceptual = lambda_perceptual
        self.lambda_geometric = lambda_geometric
        self.noise_schedule = noise_schedule
        self.r_ratio = r_ratio

        self.normalizer = MeshNormalizer(method='unit_cube')

    def sample_time(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (r, t) time pairs."""
        if self.noise_schedule == 'logit_normal':
            u = torch.randn(batch_size, device=device)
            t = torch.sigmoid(0.8 * u + 0.8)
        elif self.noise_schedule == 'uniform':
            t = torch.rand(batch_size, device=device)
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

        t = t.clamp(1e-5, 1 - 1e-5)

        use_interval = torch.rand(batch_size, device=device) < self.r_ratio
        r_uniform = t * torch.rand(batch_size, device=device)
        r = torch.where(use_interval, r_uniform, t)

        return r.unsqueeze(-1), t.unsqueeze(-1)

    def u_from_x(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert x-prediction to average velocity u.

        Args:
            z: [B, F, 3, 3] noisy face vertices
            t: [B, 1] time
            x_pred: [B, F, 3, 3] predicted face vertices

        Returns:
            u: [B, F, 3, 3] velocity
        """
        # u = (z - x) / t
        return (z - x_pred) / t.unsqueeze(-1).unsqueeze(-1)

    def compute_meanflow_loss(
        self,
        model: FaceCentricMeshMeanFlowNet,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        v_gt: torch.Tensor,
        face_mask: torch.Tensor,
        condition_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MeanFlow loss for face-centric representation.

        Args:
            z_t: [B, F, 3, 3] noisy face vertices
            r: [B, 1] start time
            t: [B, 1] end time
            v_gt: [B, F, 3, 3] ground truth velocity
            face_mask: [B, F] valid face mask
            condition_kwargs: additional condition inputs
        """
        # Get x prediction
        x_pred = model(z_t, r, t, face_mask, **condition_kwargs)

        # Convert to u
        u = self.u_from_x(z_t, t, x_pred)

        # Compute du/dt via finite difference
        eps = 1e-4
        t_plus = t + eps
        x_plus = model(z_t, r, t_plus, face_mask, **condition_kwargs)
        u_plus = self.u_from_x(z_t, t_plus, x_plus)

        dudt = (u_plus - u) / eps

        # MeanFlow compound function: V = u + (t - r) * du/dt
        # IMPORTANT: Stop gradient on du/dt
        V = u + (t - r).unsqueeze(-1).unsqueeze(-1) * dudt.detach()

        # MSE loss in velocity space
        loss = F.mse_loss(V, v_gt, reduction='none')

        # Mask invalid faces: [B, F] -> [B, F, 1, 1]
        mask = face_mask.unsqueeze(-1).unsqueeze(-1)
        loss = loss * mask
        loss = loss.sum() / (face_mask.sum() * 3 * 3)  # Normalize by valid elements

        return loss, x_pred

    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Single training step for face-centric representation.

        Args:
            batch: {
                'face_vertices': [B, F, 3, 3],  # F faces, 3 vertices per face, xyz
                'face_mask': [B, F],             # Valid face mask
                'footprint': [B, P, 2] (optional),
                'image': [B, 3, H, W] (optional),
            }
        """
        self.model.train()

        face_vertices = batch['face_vertices']  # [B, F, 3, 3] already normalized to [-1, 1]
        face_mask = batch['face_mask']          # [B, F]
        B, F, _, _ = face_vertices.shape
        device = face_vertices.device

        # Sample time
        r, t = self.sample_time(B, device)

        # Sample noise and create noisy vertices
        epsilon = torch.randn_like(face_vertices)
        z_t = (1 - t.unsqueeze(-1).unsqueeze(-1)) * face_vertices + t.unsqueeze(-1).unsqueeze(-1) * epsilon

        # Ground truth velocity
        v_gt = epsilon - face_vertices

        # Prepare condition kwargs
        condition_kwargs = {}
        if 'footprint' in batch:
            condition_kwargs['footprint'] = batch['footprint']
            condition_kwargs['footprint_mask'] = batch.get('footprint_mask')
        if 'image' in batch:
            condition_kwargs['image'] = batch['image']

        # Compute MeanFlow loss
        velocity_loss, x_pred = self.compute_meanflow_loss(
            self.model, z_t, r, t, v_gt, face_mask, condition_kwargs
        )

        total_loss = velocity_loss
        loss_dict = {'velocity_loss': velocity_loss.item()}

        # Perceptual loss (when noise is low)
        if self.perceptual_loss is not None:
            noise_mask = (t.squeeze(-1) < 0.8).float()
            if noise_mask.sum() > 0:
                perc_loss = self.perceptual_loss(
                    x_pred, face_vertices, face_mask
                )
                perc_loss = perc_loss * noise_mask.mean()

                total_loss = total_loss + self.lambda_perceptual * perc_loss
                loss_dict['perceptual_loss'] = perc_loss.item()

        # Geometric loss
        if self.geometric_loss is not None:
            geo_loss = self.geometric_loss(x_pred, face_vertices, face_mask)
            total_loss = total_loss + self.lambda_geometric * geo_loss
            loss_dict['geometric_loss'] = geo_loss.item()

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        loss_dict['total_loss'] = total_loss.item()
        return loss_dict

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        num_faces: int,
        face_mask: torch.Tensor = None,
        condition_kwargs: Dict = None,
        device: torch.device = 'cuda',
    ) -> torch.Tensor:
        """
        One-step generation (1-NFE) for face-centric representation.

        Args:
            batch_size: number of meshes to generate
            num_faces: number of faces per mesh
            face_mask: [B, F] valid face mask (optional)
            condition_kwargs: conditioning inputs

        Returns:
            x_pred: [B, F, 3, 3] generated face vertices
        """
        self.model.eval()

        if face_mask is None:
            face_mask = torch.ones(batch_size, num_faces, dtype=torch.bool, device=device)

        # Start from pure noise
        z_1 = torch.randn(batch_size, num_faces, 3, 3, device=device)
        z_1 = z_1 * face_mask.unsqueeze(-1).unsqueeze(-1)

        # One step: t=1, r=0
        t = torch.ones(batch_size, 1, device=device)
        r = torch.zeros(batch_size, 1, device=device)

        if condition_kwargs is None:
            condition_kwargs = {}

        # Predict denoised mesh
        x_pred = self.model(z_1, r, t, face_mask, **condition_kwargs)

        return x_pred

    @staticmethod
    def face_vertices_to_mesh(
        face_vertices: torch.Tensor,
        face_mask: torch.Tensor = None,
        merge_threshold: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert face-centric representation back to indexed mesh.

        This merges duplicate vertices that appear in multiple faces.

        Args:
            face_vertices: [B, F, 3, 3] or [F, 3, 3] face vertices
            face_mask: [B, F] or [F] valid face mask
            merge_threshold: distance threshold for merging vertices

        Returns:
            vertices: [B, V, 3] or [V, 3] unique vertices
            faces: [B, F, 3] or [F, 3] face indices
        """
        single_batch = face_vertices.dim() == 3
        if single_batch:
            face_vertices = face_vertices.unsqueeze(0)
            if face_mask is not None:
                face_mask = face_mask.unsqueeze(0)

        B, F, _, _ = face_vertices.shape
        device = face_vertices.device

        all_vertices = []
        all_faces = []

        for b in range(B):
            if face_mask is not None:
                valid_faces = face_mask[b].sum().item()
            else:
                valid_faces = F

            # Get valid face vertices
            fv = face_vertices[b, :valid_faces]  # [valid_F, 3, 3]

            # Flatten to all vertices
            all_verts = fv.reshape(-1, 3)  # [valid_F * 3, 3]

            # Find unique vertices by clustering close ones
            unique_verts = []
            vert_mapping = []

            for v in all_verts:
                found = False
                for i, uv in enumerate(unique_verts):
                    if torch.norm(v - uv) < merge_threshold:
                        vert_mapping.append(i)
                        found = True
                        break
                if not found:
                    vert_mapping.append(len(unique_verts))
                    unique_verts.append(v)

            vertices = torch.stack(unique_verts) if unique_verts else torch.zeros(0, 3, device=device)
            faces = torch.tensor(vert_mapping, device=device).reshape(-1, 3)

            all_vertices.append(vertices)
            all_faces.append(faces)

        if single_batch:
            return all_vertices[0], all_faces[0]

        # For batched output, would need padding - return list instead
        return all_vertices, all_faces