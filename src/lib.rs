extern crate alloc;

pub mod engine;

use bevy::app::App;
use bevy::prelude::{Plugin, Resource, Vec4};

#[derive(Resource)]
pub struct VoxelLight {
    pub ambient: Vec4,
    pub direction: Vec4,
    pub sky_color: Vec4,
}

pub struct NEVRPlugin;

impl Plugin for NEVRPlugin {
    fn build(&self, app: &mut App) {}
}
