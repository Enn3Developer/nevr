pub mod engine;

use crate::camera::{VoxelCamera, VoxelCameraData};
use crate::voxel::VoxelLibrary;
use crate::vulkan_instance::VulkanInstance;
use bevy::app::App;
use bevy::prelude::{
    Added, Changed, Commands, Entity, IntoScheduleConfigs, Plugin, Query, Res, Update,
};
pub use egui_winit_vulkano::*;
pub use engine::*;

extern crate nalgebra_glm as glm;
pub mod math {
    pub use glm::*;
}

pub mod window {
    pub use winit::*;
}

pub struct NEVRPlugin {
    pub name: String,
    pub version: Version,
}

impl NEVRPlugin {
    pub fn new(name: impl Into<String>, version: impl Into<Version>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

impl Plugin for NEVRPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(VoxelLibrary::new())
            .insert_resource(
                VulkanInstance::new(Some(self.name.clone()), self.version.clone()).unwrap(),
            )
            .add_systems(Update, (init_camera, update_camera).chain());
    }
}

fn init_camera(
    mut commands: Commands,
    camera_query: Query<(Entity, &VoxelCamera), Added<VoxelCamera>>,
    vulkan_instance: Res<VulkanInstance>,
) {
    for (entity, camera) in camera_query {
        commands
            .entity(entity)
            .insert(VoxelCameraData::new(&camera, &vulkan_instance).unwrap());
    }
}

fn update_camera(
    camera_query: Query<(&mut VoxelCameraData, &VoxelCamera), Changed<VoxelCamera>>,
    vulkan_instance: Res<VulkanInstance>,
) {
    for (camera_data, camera) in camera_query {
        // TODO:
    }
}
