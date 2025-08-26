use bevy::prelude::PerspectiveProjection;
use bevy::prelude::{Component, GlobalTransform, Projection, Transform};
use bytemuck::{Pod, Zeroable};
use std::ops::Deref;

#[derive(Clone, Debug, Component)]
#[require(
    Transform,
    Projection::Perspective(
        PerspectiveProjection {
            aspect_ratio: 16.0 / 9.0,
            fov: 90.0f32.to_radians(),
            near: 0.001,
            far: 10000.0,
        },
    ),
)]
pub struct VoxelCamera {
    pub aperture: f32,
    pub focus_distance: f32,
    pub samples: u32,
    pub bounces: u32,
}

impl VoxelCamera {
    pub fn new(aperture: f32, focus_distance: f32, samples: u32, bounces: u32) -> Self {
        Self {
            aperture,
            focus_distance,
            samples,
            bounces,
        }
    }
}

impl Default for VoxelCamera {
    fn default() -> Self {
        Self::new(0.0, 3.4, 5, 3)
    }
}
#[derive(Debug, Pod, Zeroable, Copy, Clone)]
#[repr(C)]
pub struct RayCamera {
    view_proj: [[f32; 4]; 4],
    view_inverse: [[f32; 4]; 4],
    proj_inverse: [[f32; 4]; 4],
    aperture: f32,
    focus_distance: f32,
    samples: u32,
    bounces: u32,
}

impl<
    C: Deref<Target = VoxelCamera>,
    T: Deref<Target = GlobalTransform>,
    P: Deref<Target = Projection>,
> From<(C, T, P)> for RayCamera
{
    fn from(camera_and_transform: (C, T, P)) -> Self {
        let (camera, transform, projection) = camera_and_transform;
        let projection = projection.get_clip_from_view();
        let view = transform.to_matrix();
        RayCamera {
            view_proj: (projection * view).to_cols_array_2d(),
            view_inverse: view.inverse().to_cols_array_2d(),
            proj_inverse: projection.inverse().to_cols_array_2d(),
            aperture: camera.aperture,
            focus_distance: camera.focus_distance,
            samples: camera.samples,
            bounces: camera.bounces,
        }
    }
}
