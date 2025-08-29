//! This module contains structs and traits used by [crate::engine::voxel::VoxelMaterial] for the colors.

use bevy::prelude::{IVec3, Vec3, Vec4};

/// Traits to convert data to a f32 array with length 4 used to represent an RGBA color.
pub trait IntoRgba {
    fn into_rgba(self) -> [f32; 4];
}

/// Color spaces supported by NEVR.
pub enum VoxelColor {
    /// Range: \[0, 1].
    RGBA(f32, f32, f32, f32),
}

impl VoxelColor {
    pub fn new_rgba(color: impl IntoRgba) -> Self {
        let color = color.into_rgba();
        Self::RGBA(color[0], color[1], color[2], color[3])
    }
}

impl IntoRgba for VoxelColor {
    fn into_rgba(self) -> [f32; 4] {
        match self {
            VoxelColor::RGBA(r, g, b, a) => [r, g, b, a],
        }
    }
}

impl IntoRgba for (f32, f32, f32, f32) {
    fn into_rgba(self) -> [f32; 4] {
        [self.0, self.1, self.2, self.3]
    }
}

impl IntoRgba for (f32, f32, f32) {
    fn into_rgba(self) -> [f32; 4] {
        [self.0, self.1, self.2, 1.0]
    }
}

impl IntoRgba for (f64, f64, f64, f64) {
    fn into_rgba(self) -> [f32; 4] {
        [self.0 as f32, self.1 as f32, self.2 as f32, self.3 as f32]
    }
}

impl IntoRgba for (f64, f64, f64) {
    fn into_rgba(self) -> [f32; 4] {
        [self.0 as f32, self.1 as f32, self.2 as f32, 1.0]
    }
}

impl IntoRgba for Vec4 {
    fn into_rgba(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
}

impl IntoRgba for Vec3 {
    fn into_rgba(self) -> [f32; 4] {
        [self.x, self.y, self.z, 1.0]
    }
}

impl IntoRgba for IVec3 {
    fn into_rgba(self) -> [f32; 4] {
        [
            self.x as f32 / 255.0,
            self.y as f32 / 255.0,
            self.z as f32 / 255.0,
            1.0,
        ]
    }
}
