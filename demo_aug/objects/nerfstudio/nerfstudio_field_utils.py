from functools import cached_property
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import get_normalized_directions
from torch import Tensor


def are_points_in_aabb(
    points: Tensor, min_corner: Tensor, max_corner: Tensor
) -> Tensor:
    """
    Args:
        points (torch.Tensor): points to check
        min_corner (torch.Tensor): min corner of AABB
        max_corner (torch.Tensor): max corner of AABB
    """
    return torch.all(
        torch.logical_and(min_corner <= points, points <= max_corner), dim=-1
    )


def get_outputs_tensorf_field(
    self,
    ray_samples: RaySamples,
    bounding_box_min: List[float],
    bounding_box_max: List[float],
    density_embedding: Optional[Tensor] = None,
    overall_transform_fn: Callable = lambda x: x,
    transform_params_seq: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Tensor:
    d = ray_samples.frustums.directions
    raw_positions = (
        ray_samples.frustums.get_positions()
    )  # convert to homogeneous coordinates
    raw_positions = torch.cat(
        [raw_positions, torch.ones_like(raw_positions[..., :1])], dim=-1
    )
    transformed_positions = overall_transform_fn(raw_positions)[..., :-1]

    bounding_box_min = torch.tensor(
        bounding_box_min, device=transformed_positions.device
    )
    bounding_box_max = torch.tensor(
        bounding_box_max, device=transformed_positions.device
    )

    mask = are_points_in_aabb(raw_positions, bounding_box_min, bounding_box_max)

    positions = SceneBox.get_normalized_positions(transformed_positions, self.aabb)
    positions = positions * 2 - 1

    rgb = torch.zeros(
        *positions.shape[:-1], 3, device=positions.device
    )  # Assuming rgb is a tensor with last dimension 3

    rgb_features = self.color_encoding(positions[mask])
    rgb_features = self.B(rgb_features)

    # handle case when no points are in aabb
    if mask.sum() == 0:
        return rgb

    if self.use_sh:
        sh_mult = self.sh(d[mask])[:, :, None]
        rgb_sh = rgb_features.view(
            sh_mult.shape[0], sh_mult.shape[1], 3, sh_mult.shape[-1]
        )
        rgb[mask] = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    else:
        d_encoded = self.direction_encoding(d[mask])
        rgb_features_encoded = self.feature_encoding(rgb_features)

        out = self.mlp_head(
            torch.cat([rgb_features, d[mask], rgb_features_encoded, d_encoded], dim=-1)
        )  # type: ignore
        rgb[mask] = self.field_output_rgb(out)

    return rgb


def get_density_tensorf_field(
    self,
    ray_samples: RaySamples,
    bounding_box_min: List[float],
    bounding_box_max: List[float],
    overall_transform_fn: Callable = lambda x: x,
) -> Tensor:
    # update directions as much as possible?
    raw_positions = ray_samples.frustums.get_positions()
    raw_positions = torch.cat(
        [raw_positions, torch.ones_like(raw_positions[..., :1])], dim=-1
    )
    transformed_positions = overall_transform_fn(raw_positions)[..., :-1]

    bounding_box_min = torch.tensor(
        bounding_box_min, device=transformed_positions.device
    )
    bounding_box_max = torch.tensor(
        bounding_box_max, device=transformed_positions.device
    )

    mask = are_points_in_aabb(raw_positions, bounding_box_min, bounding_box_max)

    positions = SceneBox.get_normalized_positions(transformed_positions, self.aabb)
    positions = positions * 2 - 1

    # Apply self.density_encoding only to positions inside the bounding box
    density = torch.zeros(
        *positions.shape[:-1],
        self.density_encoding(torch.zeros(3, device=transformed_positions.device)).size(
            -1
        ),
        device=positions.device,
    )

    if mask.sum() == 0:
        return torch.sum(density, dim=-1)[:, :, None]

    density[mask] = self.density_encoding(positions[mask])

    density_enc = torch.sum(density, dim=-1)[:, :, None]
    relu = torch.nn.ReLU()
    density_enc = relu(density_enc)
    return density_enc


def transform_frustums(frustums: Frustums, overall_transform_fn: Callable) -> Frustums:
    """
    Applies a transformation to the origins and directions of Frustums.

    Args:
        frustums: The original Frustums object.
        overall_transform_fn: A transformation function that operates on and returns
                              4D homogeneous coordinates.

    Returns:
        A new Frustums object with transformed get_positions() and directions.
    """
    # Transform origins to homogeneous coordinates, apply transformation, and convert back
    origins_homogeneous = torch.cat(
        [frustums.origins, torch.ones_like(frustums.origins[..., :1])], dim=-1
    )
    transformed_origins_homogeneous = overall_transform_fn(origins_homogeneous)
    transformed_origins = transformed_origins_homogeneous[..., :-1]

    # Apply the same process to a point along the direction to obtain the new directions
    direction_points = (
        frustums.origins + frustums.directions
    )  # Any point along the direction vector
    direction_points_homogeneous = torch.cat(
        [direction_points, torch.ones_like(direction_points[..., :1])], dim=-1
    )
    transformed_direction_points_homogeneous = overall_transform_fn(
        direction_points_homogeneous
    )
    transformed_direction_points = transformed_direction_points_homogeneous[..., :-1]

    # Compute new directions
    new_directions = transformed_direction_points - transformed_origins
    new_directions_normalized = new_directions / torch.norm(
        new_directions, dim=-1, keepdim=True
    )

    # Create a new Frustums object with transformed origins and directions, and other attributes set to None
    new_frustums = Frustums(
        origins=transformed_origins,
        directions=new_directions_normalized,
        starts=frustums.starts,
        ends=frustums.ends,
    )

    return new_frustums


def new_get_positions(frustums: Frustums, overall_transform_fn: Callable) -> Tensor:
    """
    Transforms the positions of the frustums in Frustums.

    Args:
        frustums: The original Frustums object.
        overall_transform_fn: A transformation function that operates on and returns
                              4D homogeneous coordinates.

    Returns:
        A tensor of transformed positions.
    """
    # manually compute the transformed positions and store them in a tensor
    if not hasattr(frustums, "newest_transformed_positions"):
        original_positions = frustums.get_positions()
        # Transform origins to homogeneous coordinates, apply transformation, and convert back
        original_positions_homogeneous = torch.cat(
            [original_positions, torch.ones_like(original_positions[..., :1])], dim=-1
        )
        transformed_origins_homogeneous = overall_transform_fn(
            original_positions_homogeneous
        )
        transformed_positions = transformed_origins_homogeneous[..., :-1]
        frustums.newest_transformed_positions = transformed_positions
    else:
        transformed_positions = frustums.newest_transformed_positions
    return transformed_positions


class RaySamplesWrapper:
    class FrustumsWrapper:
        def __init__(
            self,
            original_frustums: Frustums,
            transform_fn: Callable,
            bounding_box_min: List[float],
            bounding_box_max: List[float],
            transform_params_seq: List[Dict[str, np.ndarray]],
        ) -> None:
            """
            Note: this wrapper is hardcoded to work with the original 3D representation of the
            object.

            Args:
                original_frustums: The original Frustums object.
                transform_fn: A transformation function that operates on and returns
                              4D homogeneous coordinates.
                bounding_box_min: Minimum corner of the AABB.
                bounding_box_max: Maximum corner of the AABB.

            Further explanation on how transform_fn works:

            We take in some desired 'real world' coordinates (from RaySamples), then transform them
            to figure out which points in the original NeRF object's coordinates we should query.

            E.g., we know what p_old in the original object's coordinate system is.
            We've transformed the object with some functions T to get p_new.
            However, to render things w.r.t the original object, we need to figure out which points
            in the original object's coordinate system correspond to p_new. This is what transform_fn does.

            transform_fn = T^{-1}

            Thus, we have p_new = T @ p_old and p_old = transform_fn(p_new)

            Our bounding box is given w.r.t. the original object's coordinates. Thus,
            our masking would occur w.r.t p_old, since the mask is given w.r.t. the original
            points.
            """
            self.original_frustums = original_frustums
            self.transform_fn = transform_fn
            self._transform_params_seq = transform_params_seq
            self._bounding_box_min = (
                torch.tensor(bounding_box_min).to(original_frustums.origins)
                if not isinstance(bounding_box_min, torch.Tensor)
                else bounding_box_min
            )
            self._bounding_box_max = (
                torch.tensor(bounding_box_max).to(original_frustums.origins)
                if not isinstance(bounding_box_max, torch.Tensor)
                else bounding_box_max
            )
            self._device = original_frustums.origins.device
            self._dtype = original_frustums.origins.dtype

        @property
        def bounding_box_min(self):
            """Using property so that __getattr__ is not called when accessing this instance variable.
            Unclear why __getattr__ is called when accessing this instance variable."""
            return self._bounding_box_min

        @property
        def bounding_box_max(self):
            return self._bounding_box_max

        @property
        def directions(self):
            """Transform directions from the original frustums using transform_fn."""
            # Convert directions to homogeneous coordinates with a zero for the w component.
            directions_homo = torch.cat(
                [
                    self.original_frustums.directions,
                    torch.zeros_like(self.original_frustums.directions[..., :1]),
                ],
                dim=-1,
            )
            use_transf_params = self._transform_params_seq is not None
            if use_transf_params:
                overall_transform_matrix_expanded = (
                    self.overall_transform_matrix.unsqueeze(0).unsqueeze(0)
                )
                raw_positions_homo_expanded = directions_homo.unsqueeze(-1)
                transformed_directions_homo = torch.matmul(
                    overall_transform_matrix_expanded, raw_positions_homo_expanded
                ).squeeze(-1)
            else:
                transformed_directions_homo = self.transform_fn(directions_homo)

            return transformed_directions_homo[..., :-1]
            # TODO(klin): better implementation of directions involving the start/end points and their rotation

        @cached_property
        def overall_transform_matrix(self):
            res = torch.eye(4, device=self._device, dtype=self._dtype)
            for transform_params in self._transform_params_seq[::-1]:
                X_origin_t = next(
                    (v for k, v in transform_params.items() if "origin" in k), None
                )
                assert (
                    len([k for k in transform_params if "origin" in k]) == 1
                ), "Only one origin should be provided for a given transform"
                X_transform_t = next(
                    (v for k, v in transform_params.items() if "origin" not in k), None
                )
                assert (
                    len([k for k in transform_params if "origin" not in k]) == 1
                ), "Only one non-origin should be provided for a given transform"
                # convert to tensors
                X_origin_t = torch.tensor(
                    X_origin_t, device=self._device, dtype=self._dtype
                )
                X_transform_t = torch.tensor(
                    X_transform_t, device=self._device, dtype=self._dtype
                )
                res = torch.matmul(
                    X_origin_t,
                    torch.matmul(
                        torch.linalg.inv(X_transform_t),
                        torch.matmul(torch.linalg.inv(X_origin_t), res),
                    ),
                )
            return res

        def get_positions(self):
            """Apply transformation to positions from the original frustums."""
            raw_positions = self.original_frustums.get_positions()
            raw_positions_homo = torch.cat(
                [raw_positions, torch.ones_like(raw_positions[..., :1])], dim=-1
            )

            use_transf_params = self._transform_params_seq is not None
            if use_transf_params:
                # Adding an extra dimension to overall_transform_matrix to enable broadcasting
                overall_transform_matrix_expanded = (
                    self.overall_transform_matrix.unsqueeze(0).unsqueeze(0)
                )
                # Ensuring that the broadcasting aligns properly: [N, 1, 4, 4] @ [N, B, 4, 1]
                raw_positions_homo_expanded = raw_positions_homo.unsqueeze(-1)
                # Performing batch matrix multiplication
                transformed_positions_homo = torch.matmul(
                    overall_transform_matrix_expanded, raw_positions_homo_expanded
                ).squeeze(-1)
            else:
                transformed_positions_homo = self.transform_fn(raw_positions_homo)

            # use transform_params_seq here
            return transformed_positions_homo[..., :-1]

        @cached_property
        def bbox_mask(self):
            """Returns a mask for indices corresponding to querying points outside of the provided bbox."""
            return are_points_in_aabb(
                self.get_positions(), self.bounding_box_min, self.bounding_box_max
            )

        # other attributes should come from original_frustums
        def __getattr__(self, name: str):
            """Delegate attribute access to the original frustums if not defined in wrapper."""
            print(f"Delegating {name} to original frustums")
            # import ipdb;ipdb.set_trace
            return getattr(self.original_frustums, name)

    def __init__(
        self,
        ray_samples: RaySamples,
        transform_fn: Callable,
        bounding_box_min: List[float],
        bounding_box_max: List[float],
        transform_params_seq: Optional[List[Dict[str, np.ndarray]]] = None,
    ) -> None:
        self.original_ray_samples = ray_samples
        self.frustums = self.FrustumsWrapper(
            ray_samples.frustums,
            transform_fn,
            bounding_box_min,
            bounding_box_max,
            transform_params_seq=transform_params_seq,
        )

    def __getattr__(self, name: str):
        return getattr(self.original_ray_samples, name)


def get_density_nerfacto_field(
    self,
    ray_samples: RaySamples,
    bounding_box_min: List[float],
    bounding_box_max: List[float],
    overall_transform_fn: Callable = lambda x: x,
    transform_params_seq: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Tuple[Tensor, Tensor]:
    """Computes and returns the densities.

    Args: ray_samples: Samples locations to compute density.

    overall_transform_fn: Function to apply to the samples (transforms their positions) before computing the density.
        Usage:
        ```
        raw_positions = ray_samples.frustums.get_positions()
        raw_positions = torch.cat([raw_positions, torch.ones_like(raw_positions[..., :1])], dim=-1)
        transformed_positions = overall_transform_fn(raw_positions)[..., :-1]
        ```
    """
    ray_samples = RaySamplesWrapper(
        ray_samples,
        overall_transform_fn,
        bounding_box_min,
        bounding_box_max,
        transform_params_seq=transform_params_seq,
    )

    # I think the distortion/normalization below is fine as it expects ray_samples to be in 'absolute coordinates'
    if self.spatial_distortion is not None:
        positions = ray_samples.frustums.get_positions()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
    else:
        positions = SceneBox.get_normalized_positions(
            ray_samples.frustums.get_positions(), self.aabb
        )
    # Make sure the tcnn gets inputs between 0 and 1.
    selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
    positions = positions * selector[..., None]

    self._sample_locations = positions
    if not self._sample_locations.requires_grad:
        self._sample_locations.requires_grad = True
    positions_flat = positions.view(-1, 3)

    mask = ray_samples.frustums.bbox_mask

    def efficient_mlp(positions_flat: torch.Tensor, mask: torch.Tensor):
        # Compute mlp_base for points inside the AABB and reshape it to the required shape
        if not hasattr(self, "_mlp_base_out_dim"):
            self._mlp_base_out_dim = self.mlp_base(
                torch.zeros((1, 3), device=positions.device)
            ).size(-1)
        if not hasattr(self, "_mlp_base_out_dtype"):
            self._mlp_base_out_dtype = self.mlp_base(
                torch.zeros((1, 3), device=positions.device)
            ).dtype

        h = torch.ones(
            *ray_samples.frustums.directions.shape[:-1],
            self._mlp_base_out_dim,
            device=positions.device,
            dtype=self._mlp_base_out_dtype,
        ) * (-torch.inf)

        if mask.sum() == 0:
            return h

        mlp_base_masked = self.mlp_base(positions_flat[mask.view(-1)])
        h[mask] = mlp_base_masked
        return h

    h = efficient_mlp(positions_flat, mask)
    density_before_activation, base_mlp_out = torch.split(
        h, [1, self.geo_feat_dim], dim=-1
    )
    self._density_before_activation = density_before_activation

    # Rectifying the density with an exponential is much more stable than a ReLU or
    # softplus, because it enables high post-activation (float32) density outputs
    # from smaller internal (float16) parameters.
    density = trunc_exp(density_before_activation.to(positions))
    density = density * selector[..., None]
    return density, base_mlp_out


def get_outputs_nerfacto_field(
    self,
    ray_samples: RaySamples,
    bounding_box_min: List[float],
    bounding_box_max: List[float],
    density_embedding: Optional[Tensor] = None,
    overall_transform_fn: Callable = lambda x: x,
    transform_params_seq: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Dict[FieldHeadNames, Tensor]:
    """
    Computes and returns the colors. Returns output field values.

    Args:
        ray_samples: Samples locations to compute outputs.
        density_embedding: Density embeddings to condition on.
        overall_transform_fn: Function to apply to the samples (transforms their positions) before computing the density.
        Usage:
        ```
        raw_positions = ray_samples.frustums.get_positions()
        raw_positions = torch.cat([raw_positions, torch.ones_like(raw_positions[..., :1])], dim=-1)
        transformed_positions = overall_transform_fn(raw_positions)[..., :-1]
        ```
    """
    ray_samples = RaySamplesWrapper(
        ray_samples,
        overall_transform_fn,
        bounding_box_min,
        bounding_box_max,
        transform_params_seq,
    )

    assert density_embedding is not None
    outputs = {}
    if ray_samples.camera_indices is None:
        raise AttributeError("Camera indices are not provided.")
    camera_indices = ray_samples.camera_indices.squeeze()
    directions = get_normalized_directions(ray_samples.frustums.directions)
    directions_flat = directions.view(-1, 3)
    d = self.direction_encoding(directions_flat)

    outputs_shape = ray_samples.frustums.directions.shape[:-1]

    # appearance
    if self.training:
        embedded_appearance = self.embedding_appearance(camera_indices)
    else:
        if self.use_average_appearance_embedding:
            embedded_appearance = torch.ones(
                (*directions.shape[:-1], self.appearance_embedding_dim),
                device=directions.device,
            ) * self.embedding_appearance.mean(dim=0)
        else:
            embedded_appearance = torch.zeros(
                (*directions.shape[:-1], self.appearance_embedding_dim),
                device=directions.device,
            )

    # transients
    if self.use_transient_embedding and self.training:
        embedded_transient = self.embedding_transient(camera_indices)
        transient_input = torch.cat(
            [
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_transient.view(-1, self.transient_embedding_dim),
            ],
            dim=-1,
        )
        x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
        outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
        outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
        outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

    # semantics
    if self.use_semantics:
        semantics_input = density_embedding.view(-1, self.geo_feat_dim)
        if not self.pass_semantic_gradients:
            semantics_input = semantics_input.detach()

        x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
        outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

    # predicted normals
    if self.use_pred_normals:
        positions = ray_samples.frustums.get_positions()

        positions_flat = self.position_encoding(positions.view(-1, 3))
        pred_normals_inp = torch.cat(
            [positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1
        )

        x = (
            self.mlp_pred_normals(pred_normals_inp)
            .view(*outputs_shape, -1)
            .to(directions)
        )
        outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

    h = torch.cat(
        [
            d,
            density_embedding.view(-1, self.geo_feat_dim),
            embedded_appearance.view(-1, self.appearance_embedding_dim),
        ],
        dim=-1,
    )

    mask = ray_samples.frustums.bbox_mask

    def efficient_mlp(
        h: torch.Tensor, mask: torch.Tensor, outputs: Dict[FieldHeadNames, Tensor]
    ):
        if not hasattr(self, "_mlp_head_out_dim"):
            self._mlp_head_out_dim = self.mlp_head(
                torch.zeros((1, h.size(-1)), device=directions.device)
            ).size(-1)
        if not hasattr(self, "_mlp_head_out_dtype"):
            self._mlp_head_out_dtype = self.mlp_head(
                torch.zeros((1, h.size(-1)), device=directions.device)
            ).dtype

        rgb = torch.zeros(
            *outputs_shape,
            self._mlp_head_out_dim,
            device=directions.device,
            dtype=directions.dtype,
        )

        if mask.sum() == 0:
            return rgb

        mask_flat = mask.view(-1)
        h_flat = h.view(-1, h.size(-1))

        rgb[mask] = self.mlp_head(h_flat[mask_flat]).to(directions)
        return rgb.to(directions)

    rgb = efficient_mlp(h, mask, outputs)
    outputs.update({FieldHeadNames.RGB: rgb})

    return outputs
