//! NEVR is a voxel raytracing renderer.
//!
//! Example usage of NEVR (spawns a block composed of only one white voxel and spawns a camera to render it):
//! ```
//! use bevy::DefaultPlugins;
//! use bevy::prelude::{App, Startup, Commands, Res, AssetServer, Vec3};
//! use nevr::NEVRPlugin;
//! use nevr::engine::color::VoxelColor;
//! use nevr::engine::camera::VoxelCamera;
//! use nevr::engine::voxel::{VoxelMaterial, RelativeVoxel, VoxelType, VoxelBlock};
//!
//! fn main() {
//!     App::new()
//!         .add_plugins((DefaultPlugins, NEVRPlugin))
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//!
//! fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
//!     // create a new lambertian material with the color WHITE
//!     let material = asset_server.add(VoxelMaterial::new_lambertian(VoxelColor::RGBA(1.0, 1.0, 1.0, 1.0)));
//!     // create a list of voxels that composes a block
//!     let voxels = vec![RelativeVoxel::new(material, Vec3::ZERO)];
//!     // create a type of block which is composed of 1x1x1 voxels
//!     let voxel_type = asset_server.add(VoxelType::new(1, voxels));
//!     // spawn a new block, by default it spawns it a (0, 0, 0) with no rotation and at scale = 1
//!     // to control its position, rotation and scale use a Transform, check the VoxelBlock documentation for more information
//!     commands.spawn(VoxelBlock::new(voxel_type));
//!
//!     // spawn a new camera with default parameters
//!     // use Transform to control the position and rotation and Projection to control the projection (perspective vs orthogonal, aspect ratio, etc...)
//!     // VoxelCamera has additional parameters that you can check in the documentation
//!     commands.spawn(VoxelCamera::default());
//! }
//! ```

pub mod engine;

use crate::engine::blas::{BlasManager, compact_blas, prepare_blas};
use crate::engine::camera::{RayCamera, VoxelCamera};
use crate::engine::geometry::{GeometryManager, RenderObject, prepare_geometry, prepare_materials};
use crate::engine::light::{RenderVoxelLight, VoxelLight};
use crate::engine::node::NEVRNodeRender;
use crate::engine::voxel::{
    RenderVoxelBlock, RenderVoxelType, VoxelBlock, VoxelMaterial, VoxelType,
};
use bevy::app::App;
use bevy::prelude::{
    AssetApp, FromWorld, GlobalTransform, IntoScheduleConfigs, Mat4, Plugin, Query, Res, ResMut,
    Resource, UVec4, Vec4, World,
};
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::render::extract_resource::ExtractResourcePlugin;
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

/// Default plugin for NEVR.
///
/// Add it to your app to use NEVR:
/// ```rs
/// App::new().add_plugins((DefaultPlugins, NEVRPlugin)).run();
/// ```
///
/// **Note:** Bevy default plugins are necessary for NEVR.
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

// TODO: add better checking in the code to avoid bevy/wgpu panics to better inform users of errors in their code
impl Plugin for NEVRPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(NEVRNodeRender)
            .add_plugins(ExtractResourcePlugin::<RenderVoxelLight>::default())
            .add_plugins(RenderAssetPlugin::<VoxelMaterial>::default())
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
                (
                    prepare_materials
                        .in_set(RenderSystems::PrepareAssets)
                        .before(prepare_assets::<VoxelMaterial>),
                    prepare_geometry
                        .in_set(RenderSystems::PrepareAssets)
                        .before(prepare_assets::<RenderVoxelType>),
                )
                    .chain(),
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

/// Trait to convert data to byte slices.
pub trait ToBytes {
    /// Convert the data representation to a byte slice.
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

/// Bindings used by [engine::node::NEVRNode] for rendering purposes.
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
                            storage_buffer_read_only::<RenderObject>(false),
                            // Indices
                            storage_buffer_read_only::<UVec4>(false),
                            // Vertices
                            storage_buffer_read_only::<Vec4>(false),
                            // Normals
                            storage_buffer_read_only::<Vec4>(false),
                            // Materials
                            storage_buffer_read_only::<VoxelMaterial>(false),
                            // Material Map
                            storage_buffer_read_only::<u32>(false),
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
                            // Light and other sky parameters
                            uniform_buffer::<RenderVoxelLight>(false),
                            // View
                            uniform_buffer::<ViewUniform>(true),
                        ),
                    ),
                ),
            ],
        }
    }
}

/// Prepare bindings for rendering.
pub fn prepare_bindings(
    mut voxel_bindings: ResMut<VoxelBindings>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    blas_manager: Res<BlasManager>,
    geometry_manager: Res<GeometryManager>,
    blocks_query: Query<(&RenderVoxelBlock, &GlobalTransform)>,
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
    let mut objects = StorageBuffer::<Vec<RenderObject>>::default();

    let mut instance_id = 0;
    for (block, transform) in blocks_query {
        let voxel_type = block.voxel_type.clone();
        let Some(blas) = blas_manager.get(&block.voxel_type) else {
            continue;
        };

        let Some(id) = geometry_manager.get_object_id(&voxel_type) else {
            return;
        };

        let Some(index_id) = geometry_manager.get_index(id) else {
            return;
        };
        let Some(material_id) = geometry_manager.get_index_material(id) else {
            return;
        };

        let transform = transform.to_matrix();
        *tlas.get_mut_single(instance_id).unwrap() = Some(TlasInstance::new(
            blas,
            tlas_transform(&transform),
            instance_id as u32,
            0xFF,
        ));
        objects.get_mut().push(RenderObject {
            index: index_id,
            material_id,
        });

        instance_id += 1;
    }

    objects.write_buffer(&render_device, &render_queue);
    let Some(vertices) = geometry_manager.vertices().buffer() else {
        eprintln!("no vertices");
        return;
    };
    let Some(normals) = geometry_manager.normals().buffer() else {
        eprintln!("no normals");
        return;
    };
    let Some(indices) = geometry_manager.indices().buffer() else {
        eprintln!("no indices");
        return;
    };
    let Some(materials) = geometry_manager.materials().buffer() else {
        eprintln!("no materials");
        return;
    };
    let Some(material_map) = geometry_manager.material_map().buffer() else {
        eprintln!("no material map");
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
            materials.as_entire_binding(),
            material_map.as_entire_binding(),
        )),
    ));
}

fn tlas_transform(transform: &Mat4) -> [f32; 12] {
    transform.transpose().to_cols_array()[..12]
        .try_into()
        .unwrap()
}
