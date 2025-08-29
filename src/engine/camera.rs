//! This module contains the camera needed to render voxels for NEVR.

use crate::ToBytes;
use bevy::camera::CameraMainTextureUsages;
use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy::ecs::query::QueryItem;
use bevy::prelude::{
    Camera, Camera2d, Component, GlobalTransform, Msaa, PerspectiveProjection, Projection,
};
use bevy::render::camera::CameraRenderGraph;
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_resource::encase::internal::{
    AlignmentValue, BufferMut, SizeValue, Writer,
};
use bevy::render::render_resource::encase::private::{Metadata, WriteInto};
use bevy::render::render_resource::{ShaderType, TextureUsages};
use bevy::render::view::Hdr;
use bytemuck::{Pod, Zeroable};
use std::ops::Deref;

/// A camera to use for rendering.
///
/// This camera enables HDR automatically (check Bevy's documentation for more information about HDR).
///
/// Check the fields for more information.
#[derive(Clone, Debug, Component)]
#[require(
    Camera,
    Camera2d::default(),
    Hdr,
    Msaa::Off,
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
    /// Aperture of the camera.
    pub aperture: f32,
    /// The focus distance of the camera.
    pub focus_distance: f32,
    /// How many rays to shoot per pixel (samples per pixel).
    pub samples: u32,
    /// The maximum number of bounces per ray (used only when hitting something).
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

    pub fn with_aperture(mut self, aperture: f32) -> Self {
        self.aperture = aperture;
        self
    }

    pub fn with_focus_distance(mut self, focus_distance: f32) -> Self {
        self.focus_distance = focus_distance;
        self
    }

    pub fn with_samples(mut self, samples: u32) -> Self {
        self.samples = samples;
        self
    }

    pub fn with_bounces(mut self, bounces: u32) -> Self {
        self.bounces = bounces;
        self
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

#[derive(Debug, Pod, Zeroable, Copy, Clone, Component, Default)]
#[repr(C)]
pub struct RayCamera {
    view_proj: [f32; 16],
    view_inverse: [f32; 16],
    proj_inverse: [f32; 16],
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
            view_proj: (projection * view).to_cols_array(),
            view_inverse: view.inverse().to_cols_array(),
            proj_inverse: projection.inverse().to_cols_array(),
            aperture: camera.aperture,
            focus_distance: camera.focus_distance,
            samples: camera.samples,
            bounces: camera.bounces,
        }
    }
}

impl ShaderType for RayCamera {
    type ExtraMetadata = ();
    const METADATA: Metadata<Self::ExtraMetadata> = Metadata {
        alignment: AlignmentValue::new(16),
        has_uniform_min_alignment: false,
        min_size: SizeValue::new(320),
        is_pod: false,
        extra: (),
    };
}

impl WriteInto for RayCamera {
    fn write_into<B>(&self, writer: &mut Writer<B>)
    where
        B: BufferMut,
    {
        writer.write_slice(self.view_proj.to_bytes());
        writer.write_slice(self.view_inverse.to_bytes());
        writer.write_slice(self.proj_inverse.to_bytes());
        writer.write(&self.aperture.to_le_bytes());
        writer.write(&self.focus_distance.to_le_bytes());
        writer.write(&self.samples.to_le_bytes());
        writer.write(&self.bounces.to_le_bytes());
    }
}
