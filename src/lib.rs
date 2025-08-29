pub mod engine;

use crate::engine::blas::{BlasManager, compact_blas, prepare_blas};
use crate::engine::camera::{RayCamera, VoxelCamera};
use crate::engine::geometry::{GeometryManager, prepare_geometry};
use crate::engine::node::NEVRNodeRender;
use crate::engine::voxel::{
    RenderVoxelBlock, RenderVoxelType, VoxelBlock, VoxelMaterial, VoxelType,
};
use bevy::app::App;
use bevy::prelude::{
    AssetApp, AssetId, FromWorld, IntoScheduleConfigs, Mat4, Plugin, Query, Res, ResMut, Resource,
    Transform, Vec4, World,
};
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::{RenderAssetPlugin, prepare_assets};
use bevy::render::render_resource::binding_types::{
    acceleration_structure, storage_buffer_read_only, texture_storage_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    AccelerationStructureFlags, AccelerationStructureUpdateMode, BindGroup, BindGroupEntries,
    BindGroupLayout, BindGroupLayoutEntries, CommandEncoderDescriptor, CreateTlasDescriptor,
    ShaderStages, StorageBuffer, StorageTextureAccess, TextureFormat, TlasInstance,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::settings::WgpuFeatures;
use bevy::render::view::ViewUniform;
use bevy::render::{Render, RenderApp, RenderSystems};

/// Used for ambient light, directional light and the sky color.
///
/// Check the fields for more information.
#[derive(Resource, ExtractResource, Clone)]
pub struct VoxelLight {
    /// Ambient light, i.e. the minimum light in the scene. Defaults to 0.03
    pub ambient: Vec4,
    /// The direction for directional light. Defaults to NEG_Y, i.e. from top to bottom as the Sun in the middle of the day.
    pub direction: Vec4,
    /// The color of the sky, it's used in reflections, global illuminations, etc...
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

/// Default plugin for NEVR.
///
/// Add it to your app to use NEVR:
/// ```rs
/// App::new().add_plugins((DefaultPlugins, NEVRPlugin)).run();
/// ```
///
/// Note: Bevy default plugins are necessary for NEVR.
pub struct NEVRPlugin;

impl NEVRPlugin {
    /// Required device features to support hardware raytracing
    pub fn required_hw_features() -> WgpuFeatures {
        WgpuFeatures::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
            | WgpuFeatures::EXPERIMENTAL_RAY_QUERY
    }

    /// Required device features to support software raytracing (does not require hardware support
    /// so it can be used on older GPUs)
    pub fn required_sw_features() -> WgpuFeatures {
        todo!("Missing software raytracing")
    }
}

impl Plugin for NEVRPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(NEVRNodeRender)
            .add_plugins(ExtractResourcePlugin::<VoxelLight>::default())
            .add_plugins(RenderAssetPlugin::<RenderVoxelType>::default())
            .add_plugins(ExtractComponentPlugin::<VoxelBlock>::default())
            .add_plugins(ExtractComponentPlugin::<VoxelCamera>::default())
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
                (
                    prepare_blas
                        .after(prepare_geometry)
                        .before(prepare_assets::<RenderVoxelType>)
                        .in_set(RenderSystems::PrepareAssets),
                    compact_blas
                        .after(prepare_blas)
                        .in_set(RenderSystems::PrepareAssets),
                ),
            )
            .add_systems(
                Render,
                prepare_bindings.in_set(RenderSystems::PrepareBindGroups),
            );
    }
}

pub trait ToBytes {
    fn to_bytes(&self) -> &[u8];
}

impl ToBytes for [f32] {
    fn to_bytes(&self) -> &[u8] {
        // SAFETY: f32 always contains 4 u8
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const _, self.len() * 4) }
    }
}

impl ToBytes for [u32] {
    fn to_bytes(&self) -> &[u8] {
        // SAFETY: u32 always contains 4 u8
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
                            // TLAS
                            acceleration_structure(),
                            // Objects
                            storage_buffer_read_only::<u32>(false),
                            // Indices
                            storage_buffer_read_only::<u32>(false),
                            // Vertices
                            storage_buffer_read_only::<Vec4>(false),
                            // Normals
                            storage_buffer_read_only::<Vec4>(false),
                        ),
                    ),
                ),
                render_device.create_bind_group_layout(
                    "voxel_image_bind_group_layout",
                    &BindGroupLayoutEntries::sequential(
                        ShaderStages::COMPUTE,
                        (
                            // Camera
                            uniform_buffer::<RayCamera>(false),
                            // Texture storage view
                            texture_storage_2d(
                                TextureFormat::Rgba16Float,
                                StorageTextureAccess::WriteOnly,
                            ),
                            uniform_buffer::<ViewUniform>(true),
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
    blocks_query: Query<(&RenderVoxelBlock, &Transform)>,
) {
    voxel_bindings.bind_group = None;

    if blocks_query.is_empty() {
        eprintln!("no blocks");
        return;
    }

    let mut tlas = render_device
        .wgpu_device()
        .create_tlas(&CreateTlasDescriptor {
            label: None,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
            max_instances: blocks_query.iter().len() as u32,
        });
    let mut transforms = StorageBuffer::<Vec<Mat4>>::default();
    let mut objects = StorageBuffer::<Vec<u32>>::default();
    let mut voxel_type: AssetId<VoxelType> = AssetId::invalid();

    let mut instance_id = 0;
    for (block, transform) in blocks_query {
        voxel_type = block.voxel_type.clone();
        let Some(blas) = blas_manager.get(&block.voxel_type) else {
            continue;
        };
        // TODO: write these data to a buffer (use the VoxelType and access it through the id)
        // TODO: access to the data and convert to world position using another buffer for the transform
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
        transforms.get_mut().push(transform);
        objects.get_mut().push(instance_id as u32);

        instance_id += 1;
    }

    transforms.write_buffer(&render_device, &render_queue);
    objects.write_buffer(&render_device, &render_queue);
    let Some(vertices) = geometry_manager.get_vertices(&voxel_type) else {
        eprintln!("no vertices");
        return;
    };
    let Some(normals) = geometry_manager.get_normals(&voxel_type) else {
        eprintln!("no normals");
        return;
    };
    let Some(indices) = geometry_manager.get_indices(&voxel_type) else {
        eprintln!("no indices");
        return;
    };

    let mut command_encoder =
        render_device.create_command_encoder(&CommandEncoderDescriptor::default());
    command_encoder.build_acceleration_structures([], [&tlas]);
    render_queue.submit([command_encoder.finish()]);
    voxel_bindings.bind_group = Some(render_device.create_bind_group(
        "voxel_bindings",
        &voxel_bindings.bind_group_layouts[0],
        &BindGroupEntries::sequential((
            tlas.as_binding(),
            objects.binding().unwrap(),
            indices.as_entire_binding(),
            vertices.as_entire_binding(),
            normals.as_entire_binding(),
        )),
    ));
}

fn tlas_transform(transform: &Mat4) -> [f32; 12] {
    transform.transpose().to_cols_array()[..12]
        .try_into()
        .unwrap()
}
