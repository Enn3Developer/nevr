extern crate alloc;

pub mod engine;

use bevy::app::App;
use bevy::prelude::{Plugin, Resource, Vec4};
use bevy::render::RenderApp;
use bevy::render::renderer::RenderDevice;
use bevy::render::settings::WgpuFeatures;

#[derive(Resource)]
pub struct VoxelLight {
    pub ambient: Vec4,
    pub direction: Vec4,
    pub sky_color: Vec4,
}

pub struct NEVRPlugin;

impl NEVRPlugin {
    pub fn required_hw_features() -> WgpuFeatures {
        WgpuFeatures::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
            | WgpuFeatures::EXPERIMENTAL_RAY_QUERY
    }
    pub fn required_sw_features() -> WgpuFeatures {
        todo!("Missing software raytracing")
    }
}

impl Plugin for NEVRPlugin {
    fn build(&self, app: &mut App) {}

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app(RenderApp);
        let render_device = render_app.world().resource::<RenderDevice>();
        if !render_device
            .features()
            .contains(NEVRPlugin::required_hw_features())
        {
            eprintln!(
                "Missing features: {}\nIn the future software raytracing may be supported",
                render_device
                    .features()
                    .difference(NEVRPlugin::required_hw_features())
            );
            return;
        }
    }
}
