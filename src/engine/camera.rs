use bevy::camera::CameraMainTextureUsages;
use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy::ecs::query::QueryItem;
use bevy::prelude::{Camera, Component, GlobalTransform, PerspectiveProjection, Projection};
use bevy::render::camera::CameraRenderGraph;
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::{ShaderType, TextureUsages};
use bevy::render::view::Hdr;
use bytemuck::{Pod, Zeroable};
use std::ops::Deref;

#[derive(Clone, Debug, Component)]
#[require(
    Camera,
    Hdr,
    CameraRenderGraph::new(Core3d),
    CameraMainTextureUsages(
        TextureUsages::RENDER_ATTACHMENT
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC
        | TextureUsages::STORAGE_BINDING
    ),
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

impl ExtractComponent for VoxelCamera {
    type QueryData = (
        &'static VoxelCamera,
        &'static GlobalTransform,
        &'static Projection,
    );
    type QueryFilter = ();
    type Out = RayCamera;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(RayCamera::from(item))
    }
}

#[derive(Debug, Pod, Zeroable, Copy, Clone, ShaderType, Component)]
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
    fn from((camera, transform, projection): (C, T, P)) -> Self {
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
