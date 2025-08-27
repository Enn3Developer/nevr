pub mod engine;

use crate::engine::blas::{BlasManager, prepare_blas};
use crate::engine::camera::{RayCamera, VoxelCamera};
use crate::engine::geometry::{GeometryManager, prepare_geometry};
use crate::engine::node::{NEVRNode, NEVRNodeLabel};
use crate::engine::voxel::{
    RenderVoxelBlock, RenderVoxelType, VoxelBlock, VoxelMaterial, VoxelType,
};
use bevy::app::App;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::prelude::{
    AssetApp, FromWorld, IVec3, IntoScheduleConfigs, Mat4, Plugin, Query, Res, ResMut, Resource,
    Transform, Vec3, Vec4, World,
};
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::{RenderAssetPlugin, prepare_assets};
use bevy::render::render_graph::{RenderGraphExt, ViewNodeRunner};
use bevy::render::render_resource::binding_types::{
    acceleration_structure, storage_buffer_read_only, texture_storage_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    AccelerationStructureFlags, AccelerationStructureUpdateMode, BindGroup, BindGroupLayout,
    BindGroupLayoutEntries, CommandEncoderDescriptor, CreateTlasDescriptor, ShaderStages,
    StorageTextureAccess, TextureFormat, TlasInstance,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::settings::WgpuFeatures;
use bevy::render::{Render, RenderApp, RenderSystems};

#[derive(Resource, ExtractResource, Clone)]
pub struct VoxelLight {
    pub ambient: Vec4,
    pub direction: Vec4,
    pub sky_color: Vec4,
}

impl Default for VoxelLight {
    fn default() -> Self {
        Self {
            ambient: Vec4::new(0.03, 0.03, 0.03, 1.0),
            direction: Vec4::NEG_Y,
            sky_color: Vec4::new(0.5, 0.7, 1.0, 1.0),
        }
    }
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
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<VoxelLight>::default())
            .add_plugins(RenderAssetPlugin::<RenderVoxelType>::default())
            .add_plugins(ExtractComponentPlugin::<VoxelBlock>::extract_visible())
            .add_plugins(ExtractComponentPlugin::<VoxelCamera>::extract_visible())
            .init_asset::<VoxelMaterial>()
            .init_asset::<VoxelType>()
            .init_resource::<VoxelLight>();
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
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

        render_app
            .init_resource::<BlasManager>()
            .init_resource::<GeometryManager>()
            .init_resource::<VoxelBindings>()
            .add_systems(
                Render,
                prepare_geometry.in_set(RenderSystems::PrepareAssets),
            )
            .add_systems(
                Render,
                prepare_blas
                    .after(prepare_geometry)
                    .before(prepare_assets::<RenderVoxelType>)
                    .in_set(RenderSystems::PrepareAssets),
            )
            .add_systems(
                Render,
                prepare_bindings.in_set(RenderSystems::PrepareBindGroups),
            );

        render_app
            .add_render_graph_node::<ViewNodeRunner<NEVRNode>>(Core3d, NEVRNodeLabel)
            .add_render_graph_edges(
                Core3d,
                (Node3d::EndPrepasses, NEVRNodeLabel, Node3d::EndMainPass),
            );
    }
}

pub trait ToBytes {
    fn to_bytes(&self) -> &[u8];
}

impl ToBytes for [f32] {
    fn to_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const _, self.len() * 4) }
    }
}

impl ToBytes for [u32] {
    fn to_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const _, self.len() * 4) }
    }
}

#[derive(Resource)]
pub struct VoxelBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layouts: [BindGroupLayout; 2],
}

impl FromWorld for VoxelBindings {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        Self {
            bind_group: None,
            bind_group_layouts: [
                render_device.create_bind_group_layout(
                    "voxel_bind_group_layout",
                    &BindGroupLayoutEntries::sequential(
                        ShaderStages::COMPUTE,
                        (
                            // Camera
                            uniform_buffer::<RayCamera>(false),
                            // TLAS
                            acceleration_structure(),
                            // Objects
                            storage_buffer_read_only::<u32>(false),
                            // Indices
                            storage_buffer_read_only::<IVec3>(false),
                            // Vertices
                            storage_buffer_read_only::<Vec3>(false),
                            // Normals
                            storage_buffer_read_only::<Vec3>(false),
                        ),
                    ),
                ),
                render_device.create_bind_group_layout(
                    "voxel_image_bind_group_layout",
                    &BindGroupLayoutEntries::single(
                        ShaderStages::COMPUTE,
                        // Texture storage view
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ],
        }
    }
}

pub fn prepare_bindings(
    mut voxel_bindings: ResMut<VoxelBindings>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    blas_manager: Res<BlasManager>,
    geometry_manager: Res<GeometryManager>,
    camera_query: Query<&RayCamera>,
    blocks_query: Query<(&RenderVoxelBlock, &Transform)>,
) {
    voxel_bindings.bind_group = None;

    if blocks_query.is_empty() {
        return;
    }

    // TODO: move camera to the other bind group to enable multiple cameras
    let Ok(camera) = camera_query.single() else {
        return;
    };

    let mut tlas = render_device
        .wgpu_device()
        .create_tlas(&CreateTlasDescriptor {
            label: None,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
            max_instances: blocks_query.iter().len() as u32,
        });

    let mut instance_id = 0;
    for (block, transform) in blocks_query {
        let Some(blas) = blas_manager.get(&block.voxel_type) else {
            continue;
        };
        // TODO: write these data to a buffer
        let Some(vertices) = geometry_manager.get_vertices(&block.voxel_type) else {
            continue;
        };
        let Some(normals) = geometry_manager.get_normals(&block.voxel_type) else {
            continue;
        };
        let Some(indices) = geometry_manager.get_indices(&block.voxel_type) else {
            continue;
        };

        let transform = transform.to_matrix();
        *tlas.get_mut_single(instance_id).unwrap() = Some(TlasInstance::new(
            blas,
            tlas_transform(&transform),
            instance_id as u32,
            0xFF,
        ));

        instance_id += 1;
    }

    let mut command_encoder =
        render_device.create_command_encoder(&CommandEncoderDescriptor::default());
    command_encoder.build_acceleration_structures([], [&tlas]);
    render_queue.submit([command_encoder.finish()]);
}

fn tlas_transform(transform: &Mat4) -> [f32; 12] {
    transform.transpose().to_cols_array()[..12]
        .try_into()
        .unwrap()
}
