//! NEVR is a voxel raytracing renderer.
//!
//! Example usage of NEVR (spawns a block composed of only one white voxel and spawns a camera to render it):
//! ```
//! use bevy::DefaultPlugins;
//! use bevy::prelude::{App, Startup, Commands, Res, AssetServer, Vec3, Color};
//! use nevr::NEVRPlugin;
//! use nevr::engine::camera::VoxelCamera;
//! use nevr::engine::denoiser::VoxelDenoiser;
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
//!     // create a new white lambertian material
//! let material = asset_server.add(VoxelMaterial::new_lambertian(Color::WHITE));
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
//!
//!     // use the simple denoiser as the denoiser pipeline
//!     commands.insert_resource(VoxelDenoiser::Simple);
//! }
//! ```

pub mod engine;

use crate::engine::blas::{BlasManager, compact_blas, prepare_blas};
use crate::engine::camera::{RayCamera, VoxelCamera};
use crate::engine::denoiser::{DenoiserPlugin, VoxelDenoiser};
use crate::engine::geometry::{GeometryManager, RenderObject, prepare_geometry, prepare_materials};
use crate::engine::light::{RenderVoxelLight, VoxelLight};
use crate::engine::node::NEVRNodeRender;
use crate::engine::skybox::VoxelSkybox;
use crate::engine::voxel::{
    RenderVoxelBlock, RenderVoxelType, VoxelBlock, VoxelMaterial, VoxelType,
};
use bevy::app::App;
use bevy::diagnostic::FrameCount;
use bevy::image::ToExtents;
use bevy::prelude::{
    AssetApp, Commands, Component, DetectChanges, Entity, FromWorld, GlobalTransform,
    InheritedVisibility, IntoScheduleConfigs, Mat4, Plugin, PostUpdate, Projection, Query, Ref,
    Res, ResMut, Resource, UVec4, Vec4, With, World,
};
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_asset::{RenderAssetPlugin, prepare_assets};
use bevy::render::render_resource::binding_types::{
    acceleration_structure, sampler, storage_buffer_read_only, texture_cube, texture_storage_2d,
    uniform_buffer,
};
use bevy::render::render_resource::{
    AccelerationStructureFlags, AccelerationStructureUpdateMode, BindGroup, BindGroupEntries,
    BindGroupLayout, BindGroupLayoutEntries, CommandEncoderDescriptor, CreateTlasDescriptor,
    SamplerBindingType, ShaderStages, StorageBuffer, StorageTextureAccess, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TlasInstance,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::settings::WgpuFeatures;
use bevy::render::texture::{CachedTexture, TextureCache};
use bevy::render::view::ViewUniform;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::transform::systems::propagate_parent_transforms;

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
        app.add_plugins((NEVRNodeRender, DenoiserPlugin))
            .add_plugins(ExtractResourcePlugin::<RenderVoxelLight>::default())
            .add_plugins(ExtractResourcePlugin::<VoxelSkybox>::default())
            .add_plugins(RenderAssetPlugin::<VoxelMaterial>::default())
            .add_plugins(RenderAssetPlugin::<RenderVoxelType>::default())
            .add_plugins(ExtractComponentPlugin::<VoxelBlock>::default())
            .add_plugins(ExtractComponentPlugin::<VoxelCamera>::default())
            .init_asset::<VoxelMaterial>()
            .init_asset::<VoxelType>()
            .init_resource::<VoxelLight>()
            .add_systems(
                PostUpdate,
                reset_frame_count.after(propagate_parent_transforms),
            );
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
                prepare_view_target.in_set(RenderSystems::PrepareResources),
            )
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
    pub bind_group_layouts: [BindGroupLayout; 4],
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
                            // Accumulation Texture
                            texture_storage_2d(
                                TextureFormat::Rgba16Float,
                                StorageTextureAccess::ReadWrite,
                            ),
                        ),
                    ),
                ),
                render_device.create_bind_group_layout(
                    "voxel_g_buffer_bind_group_layout",
                    &BindGroupLayoutEntries::sequential(
                        ShaderStages::COMPUTE,
                        (
                            // Albedo
                            texture_storage_2d(
                                TextureFormat::Rgba16Float,
                                StorageTextureAccess::WriteOnly,
                            ),
                            // Normal
                            texture_storage_2d(
                                TextureFormat::Rgba16Float,
                                StorageTextureAccess::WriteOnly,
                            ),
                            // World position
                            texture_storage_2d(
                                TextureFormat::Rgba16Float,
                                StorageTextureAccess::WriteOnly,
                            ),
                        ),
                    ),
                ),
                render_device.create_bind_group_layout(
                    "voxel_skybox_bind_group_layout",
                    &BindGroupLayoutEntries::sequential(
                        ShaderStages::COMPUTE,
                        (
                            // Skybox texture
                            texture_cube(TextureSampleType::Float { filterable: true }),
                            // Sampler
                            sampler(SamplerBindingType::Filtering),
                        ),
                    ),
                ),
            ],
        }
    }
}

/// Texture view target used for rendering.
#[derive(Component)]
pub struct VoxelViewTarget {
    pub output: CachedTexture,
    pub accumulation: CachedTexture,
}

/// Texture views for g-buffer's data (used for denoising)
#[derive(Component)]
pub struct VoxelGBuffer {
    pub albedo: CachedTexture,
    pub normal: CachedTexture,
    pub world_position: CachedTexture,
    pub secondary_textures: Vec<CachedTexture>,
}

fn prepare_view_target(
    query: Query<(Entity, &ExtractedCamera), With<RayCamera>>,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    voxel_denoiser: Res<VoxelDenoiser>,
    mut commands: Commands,
) {
    for (entity, camera) in query {
        let Some(viewport) = camera.physical_viewport_size else {
            continue;
        };

        let target_descriptor = TextureDescriptor {
            label: Some("voxel_raytracing_view_target"),
            size: viewport.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        let accumulation_descriptor = TextureDescriptor {
            label: Some("voxel_raytracing_accumulation"),
            size: viewport.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        let albedo_descriptor = TextureDescriptor {
            label: Some("voxel_raytracing_albedo"),
            size: viewport.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let normal_descriptor = TextureDescriptor {
            label: Some("voxel_raytracing_normal"),
            size: viewport.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let world_position_descriptor = TextureDescriptor {
            label: Some("voxel_raytracing_world_position"),
            size: viewport.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let secondary_texture_descriptor = TextureDescriptor {
            label: Some("voxel_raytracing_a_trous_secondary_texture"),
            size: viewport.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        let secondary_textures = if let VoxelDenoiser::ATrous(size) = *voxel_denoiser {
            let size = (size.get() as f32).log2().floor() as usize + 1;
            let mut textures = Vec::with_capacity(size);

            for _ in 0..size {
                textures
                    .push(texture_cache.get(&render_device, secondary_texture_descriptor.clone()));
            }

            textures
        } else {
            vec![]
        };

        commands
            .entity(entity)
            .insert(VoxelViewTarget {
                output: texture_cache.get(&render_device, target_descriptor),
                accumulation: texture_cache.get(&render_device, accumulation_descriptor),
            })
            .insert(VoxelGBuffer {
                albedo: texture_cache.get(&render_device, albedo_descriptor),
                normal: texture_cache.get(&render_device, normal_descriptor),
                world_position: texture_cache.get(&render_device, world_position_descriptor),
                secondary_textures,
            });
    }
}

// TODO: prepare only the changed blocks instead of preparing every block
/// Prepare bindings for rendering.
pub fn prepare_bindings(
    mut voxel_bindings: ResMut<VoxelBindings>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    blas_manager: Res<BlasManager>,
    geometry_manager: Res<GeometryManager>,
    blocks_query: Query<(&RenderVoxelBlock, &GlobalTransform, &InheritedVisibility)>,
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
    for (block, transform, visible) in blocks_query {
        if *visible == InheritedVisibility::HIDDEN {
            continue;
        }
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

pub fn reset_frame_count(
    camera_query: Query<
        (Ref<VoxelCamera>, Ref<GlobalTransform>, Ref<Projection>),
        With<VoxelCamera>,
    >,
    mut frame_count: ResMut<FrameCount>,
) {
    let mut changed = false;

    for (camera, transform, projection) in camera_query.iter() {
        changed = camera.is_changed() || transform.is_changed() || projection.is_changed();
    }

    if changed {
        frame_count.0 = u32::MAX;
    }
}

fn tlas_transform(transform: &Mat4) -> [f32; 12] {
    transform.transpose().to_cols_array()[..12]
        .try_into()
        .unwrap()
}
