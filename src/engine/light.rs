//! This module contains resources and systems to manage directional lights and various other atmosphere effects
//! like sky color and ambient light

use crate::ToBytes;
use bevy::math::Vec4;
use bevy::prelude::Resource;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::ShaderType;
use bevy::render::render_resource::encase::internal::{
    AlignmentValue, BufferMut, WriteInto, Writer,
};
use bevy::render::render_resource::encase::private::{Metadata, SizeValue};

/// Used for ambient light, directional light and its intensity, and the sky color.
#[derive(Resource, Clone)]
pub struct VoxelLight {
    /// Ambient light and light intensity, i.e. the minimum light in the scene. Defaults to (0.03, 1.0)
    pub(crate) ambient: Vec4,
    /// The direction for directional light. Defaults to NEG_Y, i.e. from top to bottom as the Sun in the middle of the day.
    pub(crate) direction: Vec4,
    /// The color of the sky, it's used in reflections, global illuminations, etc...
    pub(crate) sky_color: Vec4,
}

impl VoxelLight {
    pub fn ambient(&self) -> f32 {
        self.ambient.x
    }

    pub fn intensity(&self) -> f32 {
        self.ambient.y
    }

    pub fn direction(&self) -> Vec4 {
        self.direction
    }

    pub fn sky_color(&self) -> Vec4 {
        self.sky_color
    }

    pub fn set_ambient(&mut self, ambient_light: f32) {
        self.ambient.x = ambient_light;
    }

    pub fn set_intensity(&mut self, light_intensity: f32) {
        self.ambient.y = light_intensity;
    }

    pub fn set_direction(&mut self, direction: Vec4) {
        self.direction = direction;
    }

    pub fn set_sky_color(&mut self, sky_color: Vec4) {
        self.sky_color = sky_color;
    }
}

impl Default for VoxelLight {
    fn default() -> Self {
        Self {
            ambient: Vec4::new(0.03, 1.0, 0.03, 1.0),
            direction: Vec4::NEG_Y,
            sky_color: Vec4::new(0.5, 0.7, 1.0, 1.0),
        }
    }
}

#[derive(Resource, Default)]
pub struct RenderVoxelLight {
    pub ambient: [f32; 4],
    pub direction: [f32; 4],
    pub sky_color: [f32; 4],
}

impl ExtractResource for RenderVoxelLight {
    type Source = VoxelLight;

    fn extract_resource(source: &Self::Source) -> Self {
        Self {
            ambient: source.ambient.to_array(),
            direction: source.direction.to_array(),
            sky_color: source.sky_color.to_array(),
        }
    }
}

impl ShaderType for RenderVoxelLight {
    type ExtraMetadata = ();
    const METADATA: Metadata<Self::ExtraMetadata> = Metadata {
        alignment: AlignmentValue::new(16),
        has_uniform_min_alignment: false,
        min_size: SizeValue::new(48),
        is_pod: false,
        extra: (),
    };
}

impl WriteInto for RenderVoxelLight {
    fn write_into<B>(&self, writer: &mut Writer<B>)
    where
        B: BufferMut,
    {
        writer.write_slice(self.ambient.to_bytes());
        writer.write_slice(self.direction.to_bytes());
        writer.write_slice(self.sky_color.to_bytes());
    }
}
