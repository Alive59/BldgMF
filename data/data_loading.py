"""
Data loading utilities for BldgMF mesh generation.

Provides Dataset and DataLoader implementations for training
the Mesh MeanFlow model.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class MeshDataset(Dataset):
    """
    Dataset for loading 3D building meshes with optional conditioning.

    Supports:
    - OBJ, PLY, OFF mesh formats (via trimesh)
    - Optional 2D footprint polygons
    - Optional conditioning images (satellite/aerial)

    Expected directory structure:
        data_root/
        ├── meshes/
        │   ├── building_001.obj
        │   ├── building_002.obj
        │   └── ...
        ├── footprints/  (optional)
        │   ├── building_001.json  # {"points": [[x1,y1], [x2,y2], ...]}
        │   └── ...
        └── images/  (optional)
            ├── building_001.png
            └── ...
    """

    def __init__(
        self,
        data_root: str,
        max_vertices: int = 1024,
        max_faces: int = 2048,
        max_footprint_points: int = 64,
        use_footprint: bool = True,
        use_image: bool = False,
        image_size: int = 224,
        mesh_extensions: List[str] = None,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        augment: bool = True,
    ):
        """
        Args:
            data_root: Root directory containing meshes/, footprints/, images/
            max_vertices: Maximum number of vertices (pad/truncate to this)
            max_faces: Maximum number of faces
            max_footprint_points: Maximum footprint polygon points
            use_footprint: Whether to load footprint conditioning
            use_image: Whether to load image conditioning
            image_size: Size to resize conditioning images
            mesh_extensions: File extensions to consider as meshes
            split: 'train', 'val', or 'test'
            split_ratio: Train/val/test ratios
            seed: Random seed for splitting
            augment: Whether to apply data augmentation (train only)
        """
        super().__init__()

        if mesh_extensions is None:
            mesh_extensions = ['.obj', '.ply', '.off']

        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for MeshDataset. Install with: pip install trimesh")

        self.data_root = Path(data_root)
        self.max_vertices = max_vertices
        self.max_faces = max_faces
        self.max_footprint_points = max_footprint_points
        self.use_footprint = use_footprint
        self.use_image = use_image
        self.image_size = image_size
        self.split = split
        self.augment = augment and (split == 'train')

        # Find all mesh files
        mesh_dir = self.data_root / 'meshes'
        if not mesh_dir.exists():
            raise ValueError(f"Mesh directory not found: {mesh_dir}")

        all_files = []
        for ext in mesh_extensions:
            all_files.extend(mesh_dir.glob(f'*{ext}'))
        all_files = sorted(all_files)

        if len(all_files) == 0:
            raise ValueError(f"No mesh files found in {mesh_dir}")

        # Split dataset
        np.random.seed(seed)
        indices = np.random.permutation(len(all_files))

        train_end = int(len(all_files) * split_ratio[0])
        val_end = train_end + int(len(all_files) * split_ratio[1])

        if split == 'train':
            split_indices = indices[:train_end]
        elif split == 'val':
            split_indices = indices[train_end:val_end]
        else:  # test
            split_indices = indices[val_end:]

        self.mesh_files = [all_files[i] for i in split_indices]

        print(f"MeshDataset [{split}]: {len(self.mesh_files)} samples")

    def __len__(self) -> int:
        return len(self.mesh_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mesh_path = self.mesh_files[idx]
        sample_name = mesh_path.stem

        # Load mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        vertices, faces = self._process_mesh(mesh)

        # Create masks
        num_vertices = min(vertices.shape[0], self.max_vertices)
        num_faces = min(faces.shape[0], self.max_faces)

        vertex_mask = torch.zeros(self.max_vertices, dtype=torch.bool)
        vertex_mask[:num_vertices] = True

        face_mask = torch.zeros(self.max_faces, dtype=torch.bool)
        face_mask[:num_faces] = True

        # Pad vertices and faces
        padded_vertices = torch.zeros(self.max_vertices, 3, dtype=torch.float32)
        padded_vertices[:num_vertices] = torch.from_numpy(vertices[:num_vertices]).float()

        padded_faces = torch.zeros(self.max_faces, 3, dtype=torch.long)
        padded_faces[:num_faces] = torch.from_numpy(faces[:num_faces]).long()

        # Apply augmentation
        if self.augment:
            padded_vertices, padded_faces = self._augment(padded_vertices, padded_faces, vertex_mask)

        batch = {
            'vertices': padded_vertices,
            'faces': padded_faces,
            'vertex_mask': vertex_mask,
            'face_mask': face_mask,
            'name': sample_name,
        }

        # Load footprint if available
        if self.use_footprint:
            footprint, footprint_mask = self._load_footprint(sample_name)
            batch['footprint'] = footprint
            batch['footprint_mask'] = footprint_mask

        # Load image if available
        if self.use_image:
            image = self._load_image(sample_name)
            if image is not None:
                batch['image'] = image

        return batch

    def _process_mesh(self, mesh: 'trimesh.Trimesh') -> Tuple[np.ndarray, np.ndarray]:
        """Process and simplify mesh if needed."""
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)

        # Simplify if too many vertices/faces
        if len(vertices) > self.max_vertices or len(faces) > self.max_faces:
            target_faces = min(self.max_faces, len(faces))
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
                vertices = np.array(mesh.vertices, dtype=np.float32)
                faces = np.array(mesh.faces, dtype=np.int64)
            except Exception:
                # Fallback: random sampling
                if len(vertices) > self.max_vertices:
                    indices = np.random.choice(len(vertices), self.max_vertices, replace=False)
                    vertices = vertices[indices]
                    # Remap faces (this may invalidate some faces)
                    index_map = {old: new for new, old in enumerate(indices)}
                    valid_faces = []
                    for f in faces:
                        if all(v in index_map for v in f):
                            valid_faces.append([index_map[v] for v in f])
                    faces = np.array(valid_faces[:self.max_faces], dtype=np.int64)

        return vertices, faces

    def _augment(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentation to mesh."""
        # Random rotation around Y-axis (up)
        if np.random.rand() < 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=vertices.dtype)
            vertices[mask] = vertices[mask] @ rot_matrix.T

        # Random scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            vertices[mask] = vertices[mask] * scale

        # Random translation (small)
        if np.random.rand() < 0.5:
            translation = torch.randn(3) * 0.1
            vertices[mask] = vertices[mask] + translation

        # Random flip (X-axis)
        if np.random.rand() < 0.5:
            vertices[mask, 0] = -vertices[mask, 0]
            # Flip face winding
            faces = faces[:, [0, 2, 1]]

        return vertices, faces

    def _load_footprint(self, sample_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load footprint polygon for a sample."""
        footprint_dir = self.data_root / 'footprints'
        footprint_path = footprint_dir / f'{sample_name}.json'

        footprint = torch.zeros(self.max_footprint_points, 2, dtype=torch.float32)
        footprint_mask = torch.zeros(self.max_footprint_points, dtype=torch.bool)

        if footprint_path.exists():
            with open(footprint_path, 'r') as f:
                data = json.load(f)

            points = np.array(data.get('points', data.get('polygon', [])), dtype=np.float32)

            if len(points) > 0:
                # Normalize footprint to [-1, 1]
                points = points - points.mean(axis=0, keepdims=True)
                max_extent = np.abs(points).max()
                if max_extent > 0:
                    points = points / max_extent

                num_points = min(len(points), self.max_footprint_points)
                footprint[:num_points] = torch.from_numpy(points[:num_points])
                footprint_mask[:num_points] = True

        return footprint, footprint_mask

    def _load_image(self, sample_name: str) -> Optional[torch.Tensor]:
        """Load conditioning image for a sample."""
        if not PIL_AVAILABLE:
            return None

        image_dir = self.data_root / 'images'

        # Try common extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            image_path = image_dir / f'{sample_name}{ext}'
            if image_path.exists():
                img = Image.open(image_path).convert('RGB')
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                img = np.array(img, dtype=np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
                return img

        return None


class SyntheticMeshDataset(Dataset):
    """
    Synthetic dataset for testing/debugging.
    Generates random box-like meshes with optional footprints.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        max_vertices: int = 1024,
        max_faces: int = 2048,
        max_footprint_points: int = 64,
        use_footprint: bool = True,
        use_image: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_vertices = max_vertices
        self.max_faces = max_faces
        self.max_footprint_points = max_footprint_points
        self.use_footprint = use_footprint
        self.use_image = use_image

        np.random.seed(seed)
        self._generate_samples()

    def _generate_samples(self):
        """Pre-generate synthetic samples."""
        self.samples = []

        for _ in range(self.num_samples):
            # Random box dimensions
            width = np.random.uniform(0.5, 2.0)
            depth = np.random.uniform(0.5, 2.0)
            height = np.random.uniform(1.0, 5.0)

            # Box vertices (8 vertices)
            vertices = np.array([
                [0, 0, 0], [width, 0, 0], [width, depth, 0], [0, depth, 0],
                [0, 0, height], [width, 0, height], [width, depth, height], [0, depth, height],
            ], dtype=np.float32)

            # Center the box
            vertices -= vertices.mean(axis=0, keepdims=True)

            # Box faces (12 triangles)
            faces = np.array([
                [0, 1, 2], [0, 2, 3],  # Bottom
                [4, 6, 5], [4, 7, 6],  # Top
                [0, 4, 5], [0, 5, 1],  # Front
                [2, 6, 7], [2, 7, 3],  # Back
                [0, 3, 7], [0, 7, 4],  # Left
                [1, 5, 6], [1, 6, 2],  # Right
            ], dtype=np.int64)

            # Footprint (bottom face outline)
            footprint = np.array([
                [0, 0], [width, 0], [width, depth], [0, depth]
            ], dtype=np.float32)
            footprint -= footprint.mean(axis=0, keepdims=True)
            max_extent = np.abs(footprint).max()
            if max_extent > 0:
                footprint /= max_extent

            self.samples.append({
                'vertices': vertices,
                'faces': faces,
                'footprint': footprint,
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Pad vertices
        vertices = torch.zeros(self.max_vertices, 3, dtype=torch.float32)
        num_v = len(sample['vertices'])
        vertices[:num_v] = torch.from_numpy(sample['vertices'])

        vertex_mask = torch.zeros(self.max_vertices, dtype=torch.bool)
        vertex_mask[:num_v] = True

        # Pad faces
        faces = torch.zeros(self.max_faces, 3, dtype=torch.long)
        num_f = len(sample['faces'])
        faces[:num_f] = torch.from_numpy(sample['faces'])

        face_mask = torch.zeros(self.max_faces, dtype=torch.bool)
        face_mask[:num_f] = True

        batch = {
            'vertices': vertices,
            'faces': faces,
            'vertex_mask': vertex_mask,
            'face_mask': face_mask,
        }

        if self.use_footprint:
            footprint = torch.zeros(self.max_footprint_points, 2, dtype=torch.float32)
            num_fp = len(sample['footprint'])
            footprint[:num_fp] = torch.from_numpy(sample['footprint'])

            footprint_mask = torch.zeros(self.max_footprint_points, dtype=torch.bool)
            footprint_mask[:num_fp] = True

            batch['footprint'] = footprint
            batch['footprint_mask'] = footprint_mask

        if self.use_image:
            # Random noise image for testing
            batch['image'] = torch.randn(3, 224, 224)

        return batch


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with appropriate settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=mesh_collate_fn,
    )


def mesh_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for mesh batches."""
    collated = {}

    for key in batch[0].keys():
        if key == 'name':
            collated[key] = [b[key] for b in batch]
        elif batch[0][key] is not None:
            collated[key] = torch.stack([b[key] for b in batch], dim=0)

    return collated


# Convenience functions for quick setup
def get_train_val_loaders(
    data_root: str,
    batch_size: int = 8,
    max_vertices: int = 1024,
    use_footprint: bool = True,
    use_image: bool = False,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Get train and validation data loaders."""
    train_dataset = MeshDataset(
        data_root=data_root,
        max_vertices=max_vertices,
        use_footprint=use_footprint,
        use_image=use_image,
        split='train',
        augment=True,
    )

    val_dataset = MeshDataset(
        data_root=data_root,
        max_vertices=max_vertices,
        use_footprint=use_footprint,
        use_image=use_image,
        split='val',
        augment=False,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader


def get_synthetic_loader(
    num_samples: int = 1000,
    batch_size: int = 8,
    max_vertices: int = 1024,
    use_footprint: bool = True,
) -> DataLoader:
    """Get a synthetic data loader for testing."""
    dataset = SyntheticMeshDataset(
        num_samples=num_samples,
        max_vertices=max_vertices,
        use_footprint=use_footprint,
    )

    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Synthetic data is fast, no need for workers
    )


# ============================================================================
# Face-Centric Dataset Classes
# ============================================================================


class FaceCentricMeshDataset(Dataset):
    """
    Face-centric dataset for BldgMF Kyoto building meshes.

    Data structure:
        mesh_root/{tile_id}/{bldg_id}.obj
        image_root/{tile_id}/{bldg_id}.tif
        footprint_root/{tile_id}.geojson  (contains all bldg_ids for that tile)

    Each building is identified by bldg_id (e.g., bldg_b1c9f266-0be8-4a2d-...).
    OBJ vertices are in world coordinates (EPSG:30169) and need centering.
    Footprints are MultiPolygon in the same CRS and also need centering.
    """

    def __init__(
        self,
        data_root: str = None,
        mesh_root: str = None,
        image_root: str = None,
        footprint_root: str = None,
        max_faces: int = 256,
        max_footprint_points: int = 64,
        use_footprint: bool = True,
        use_image: bool = False,
        image_size: int = 224,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        augment: bool = True,
    ):
        super().__init__()

        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required. Install with: pip install trimesh")

        # Resolve paths
        if data_root is not None:
            data_root = Path(data_root)
            if mesh_root is None:
                mesh_root = data_root / 'obj_unlabeled_2025' / '100k' / '2022' / 'kyoto'
            if image_root is None:
                image_root = data_root / 'tiff_kyoto_buf1_2025'
            if footprint_root is None:
                footprint_root = data_root / 'geojson_kyoto_2025'

        self.mesh_root = Path(mesh_root)
        self.image_root = Path(image_root) if image_root else None
        self.footprint_root = Path(footprint_root) if footprint_root else None
        self.max_faces = max_faces
        self.max_footprint_points = max_footprint_points
        self.use_footprint = use_footprint
        self.use_image = use_image
        self.image_size = image_size
        self.split = split
        self.augment = augment and (split == 'train')

        # Build index: list of (tile_id, bldg_id, obj_path)
        self.samples = self._build_index()

        # Split dataset
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))

        train_end = int(len(self.samples) * split_ratio[0])
        val_end = train_end + int(len(self.samples) * split_ratio[1])

        if split == 'train':
            split_indices = indices[:train_end]
        elif split == 'val':
            split_indices = indices[train_end:val_end]
        else:  # test
            split_indices = indices[val_end:]

        self.samples = [self.samples[i] for i in split_indices]

        # Pre-load footprint geojsons (indexed by tile_id -> bldg_id -> feature)
        self.footprint_cache = {}
        if self.use_footprint and self.footprint_root is not None:
            self._preload_footprints()

        print(f"FaceCentricMeshDataset [{split}]: {len(self.samples)} samples")

    def _build_index(self) -> List[Tuple[str, str, Path]]:
        """Scan mesh_root for all OBJ files and build (tile_id, bldg_id, path) index."""
        samples = []
        for tile_dir in sorted(self.mesh_root.iterdir()):
            if not tile_dir.is_dir():
                continue
            tile_id = tile_dir.name
            for obj_file in sorted(tile_dir.glob('*.obj')):
                bldg_id = obj_file.stem  # e.g., bldg_b1c9f266-0be8-4a2d-...
                samples.append((tile_id, bldg_id, obj_file))
        return samples

    def _preload_footprints(self):
        """Load all geojson files and index features by bldg_id."""
        for geojson_file in sorted(self.footprint_root.glob('*.geojson')):
            tile_id = geojson_file.stem
            with open(geojson_file, 'r') as f:
                data = json.load(f)

            tile_cache = {}
            for feature in data.get('features', []):
                bldg_id = feature['properties'].get('id', '')
                tile_cache[bldg_id] = feature
            self.footprint_cache[tile_id] = tile_cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tile_id, bldg_id, obj_path = self.samples[idx]

        # Load mesh and convert to face-centric
        mesh = trimesh.load(str(obj_path), force='mesh')
        face_vertices = self._process_mesh(mesh)

        # Create mask and pad
        num_faces = min(face_vertices.shape[0], self.max_faces)
        face_mask = torch.zeros(self.max_faces, dtype=torch.bool)
        face_mask[:num_faces] = True

        padded = torch.zeros(self.max_faces, 3, 3, dtype=torch.float32)
        padded[:num_faces] = torch.from_numpy(face_vertices[:num_faces]).float()

        # Apply augmentation
        if self.augment:
            padded = self._augment(padded, face_mask)

        batch = {
            'face_vertices': padded,   # [max_faces, 3, 3]
            'face_mask': face_mask,     # [max_faces]
            'name': bldg_id,
        }

        # Load footprint
        if self.use_footprint:
            footprint, footprint_mask = self._load_footprint(tile_id, bldg_id)
            batch['footprint'] = footprint
            batch['footprint_mask'] = footprint_mask

        # Load image
        if self.use_image:
            image = self._load_image(tile_id, bldg_id)
            if image is not None:
                batch['image'] = image

        return batch

    def _process_mesh(self, mesh: 'trimesh.Trimesh') -> np.ndarray:
        """
        Convert mesh to face-centric [F, 3, 3], normalized to [-1, 1].

        OBJ vertices are in world coordinates (EPSG:30169, large values).
        We center per-building and scale to [-1, 1] (unit cube).
        """
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)

        # Simplify if too many faces (before normalization to keep topology)
        if len(faces) > self.max_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(self.max_faces)
                vertices = np.array(mesh.vertices, dtype=np.float32)
                faces = np.array(mesh.faces, dtype=np.int64)
            except Exception:
                faces = faces[:self.max_faces]

        # Normalize to [-1, 1] unit cube
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        center = (v_min + v_max) / 2
        scale = (v_max - v_min).max() / 2
        if scale < 1e-6:
            scale = 1.0
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -1.0, 1.0)

        # Convert to face-centric
        face_vertices = vertices[faces]  # [F, 3, 3]
        return face_vertices

    def _augment(
        self,
        face_vertices: torch.Tensor,
        face_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply random augmentation to face-centric mesh."""
        F, V_per_face, coords = face_vertices.shape
        flat = face_vertices.reshape(-1, 3)  # [F*3, 3]
        flat_mask = face_mask.unsqueeze(-1).expand(-1, 3).reshape(-1)  # [F*3]

        # Random rotation around Y-axis (height axis)
        if np.random.rand() < 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=flat.dtype)
            flat[flat_mask] = flat[flat_mask] @ rot_matrix.T

        # Random scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            flat[flat_mask] = flat[flat_mask] * scale

        # Random flip (X-axis)
        if np.random.rand() < 0.5:
            flat[flat_mask, 0] = -flat[flat_mask, 0]
            face_vertices = flat.reshape(F, V_per_face, coords)
            # Flip winding order
            face_vertices = face_vertices[:, [0, 2, 1], :]
            return face_vertices

        return flat.reshape(F, V_per_face, coords)

    def _load_footprint(
        self, tile_id: str, bldg_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load footprint polygon from cached geojson.

        Extracts 2D polygon coordinates, centers at origin, normalizes to [-1, 1].
        """
        footprint = torch.zeros(self.max_footprint_points, 2, dtype=torch.float32)
        footprint_mask = torch.zeros(self.max_footprint_points, dtype=torch.bool)

        tile_cache = self.footprint_cache.get(tile_id, {})
        feature = tile_cache.get(bldg_id)

        if feature is None:
            return footprint, footprint_mask

        # Extract polygon coordinates from MultiPolygon
        geometry = feature.get('geometry', {})
        coords_list = geometry.get('coordinates', [])

        if len(coords_list) == 0:
            return footprint, footprint_mask

        # Take the first polygon, first ring (outer boundary)
        # MultiPolygon: [polygon][ring][point][xy]
        ring = coords_list[0][0]  # Outer ring of first polygon
        points = np.array(ring, dtype=np.float64)

        # Remove closing point if it duplicates the first
        if len(points) > 1 and np.allclose(points[0], points[-1]):
            points = points[:-1]

        if len(points) == 0:
            return footprint, footprint_mask

        # Use only X and Z (horizontal plane) - matching OBJ axes
        # GeoJSON coords are [x, y] in EPSG:30169
        points_2d = points[:, :2].astype(np.float32)

        # Center at origin
        points_2d = points_2d - points_2d.mean(axis=0, keepdims=True)

        # Normalize to [-1, 1]
        max_extent = np.abs(points_2d).max()
        if max_extent > 0:
            points_2d = points_2d / max_extent

        num_points = min(len(points_2d), self.max_footprint_points)
        footprint[:num_points] = torch.from_numpy(points_2d[:num_points])
        footprint_mask[:num_points] = True

        return footprint, footprint_mask

    def _load_image(self, tile_id: str, bldg_id: str) -> Optional[torch.Tensor]:
        """Load conditioning TIFF image."""
        if not PIL_AVAILABLE or self.image_root is None:
            return None

        image_path = self.image_root / tile_id / f'{bldg_id}.tif'

        if not image_path.exists():
            return None

        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
        return img


class SyntheticFaceCentricMeshDataset(Dataset):
    """
    Synthetic face-centric dataset for testing/debugging.
    Generates random box-like meshes in face-centric format.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        max_faces: int = 256,
        max_footprint_points: int = 64,
        use_footprint: bool = True,
        use_image: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_faces = max_faces
        self.max_footprint_points = max_footprint_points
        self.use_footprint = use_footprint
        self.use_image = use_image

        np.random.seed(seed)
        self._generate_samples()

    def _generate_samples(self):
        """Pre-generate synthetic samples in face-centric format."""
        self.samples = []

        for _ in range(self.num_samples):
            # Random box dimensions
            width = np.random.uniform(0.5, 2.0)
            depth = np.random.uniform(0.5, 2.0)
            height = np.random.uniform(1.0, 5.0)

            # Box vertices (8 vertices)
            vertices = np.array([
                [0, 0, 0], [width, 0, 0], [width, depth, 0], [0, depth, 0],
                [0, 0, height], [width, 0, height], [width, depth, height], [0, depth, height],
            ], dtype=np.float32)

            # Center the box
            vertices -= vertices.mean(axis=0, keepdims=True)

            # Box faces (12 triangles)
            faces = np.array([
                [0, 1, 2], [0, 2, 3],  # Bottom
                [4, 6, 5], [4, 7, 6],  # Top
                [0, 4, 5], [0, 5, 1],  # Front
                [2, 6, 7], [2, 7, 3],  # Back
                [0, 3, 7], [0, 7, 4],  # Left
                [1, 5, 6], [1, 6, 2],  # Right
            ], dtype=np.int64)

            # Convert to face-centric: [12, 3, 3]
            face_vertices = vertices[faces]

            # Footprint (bottom face outline)
            footprint = np.array([
                [0, 0], [width, 0], [width, depth], [0, depth]
            ], dtype=np.float32)
            footprint -= footprint.mean(axis=0, keepdims=True)
            max_extent = np.abs(footprint).max()
            if max_extent > 0:
                footprint /= max_extent

            self.samples.append({
                'face_vertices': face_vertices,
                'footprint': footprint,
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Pad face_vertices
        face_vertices = torch.zeros(self.max_faces, 3, 3, dtype=torch.float32)
        num_f = len(sample['face_vertices'])
        face_vertices[:num_f] = torch.from_numpy(sample['face_vertices'])

        face_mask = torch.zeros(self.max_faces, dtype=torch.bool)
        face_mask[:num_f] = True

        batch = {
            'face_vertices': face_vertices,  # [F, 3, 3]
            'face_mask': face_mask,           # [F]
        }

        if self.use_footprint:
            footprint = torch.zeros(self.max_footprint_points, 2, dtype=torch.float32)
            num_fp = len(sample['footprint'])
            footprint[:num_fp] = torch.from_numpy(sample['footprint'])

            footprint_mask = torch.zeros(self.max_footprint_points, dtype=torch.bool)
            footprint_mask[:num_fp] = True

            batch['footprint'] = footprint
            batch['footprint_mask'] = footprint_mask

        if self.use_image:
            batch['image'] = torch.randn(3, 224, 224)

        return batch
