pub mod engine;

use crate::engine::camera::{VoxelCamera, VoxelCameraData};
use crate::engine::voxel::{VoxelBlock, VoxelLibrary};
use crate::engine::vulkan_instance::VulkanInstance;
use crate::engine::world::VoxelWorld;
use bevy::app::{App, MainScheduleOrder};
use bevy::asset::RenderAssetUsages;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::{
    AssetServer, Assets, Camera2d, Changed, Commands, DetectChanges, Entity, EventReader,
    GlobalTransform, Handle, IntoScheduleConfigs, Or, Plugin, PostUpdate, Query, Ref, Res, ResMut,
    Resource, Sprite, Startup, Update, Vec3, Vec4,
};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::WindowResized;
use itertools::Itertools;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
    PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::GpuFuture;
use vulkano::{DeviceSize, Version};

#[derive(Debug, BufferContents, Copy, Clone)]
#[repr(C)]
pub struct Light {
    pub(crate) ambient_light: [f32; 4],
    pub(crate) light_direction: [f32; 4],
}

pub struct NEVRPlugin {
    pub name: String,
    pub version: Version,
}

#[derive(Resource)]
pub struct VoxelLight {
    pub ambient: Vec4,
    pub direction: Vec4,
    pub sky_color: Vec3,
}

#[derive(Resource, Default)]
pub struct DescriptorSets {
    pub descriptor_set: Option<Arc<DescriptorSet>>,
    pub intersect_descriptor_set: Option<Arc<DescriptorSet>>,
    pub sky_color_descriptor_set: Option<Arc<DescriptorSet>>,
    pub image_descriptor_sets: [Option<(Arc<ImageView>, Arc<DescriptorSet>)>; 3],
}

#[derive(Resource)]
pub struct VoxelRenderTarget(Handle<bevy::image::Image>);

impl NEVRPlugin {
    pub fn new(name: impl Into<String>, version: impl Into<Version>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

#[derive(ScheduleLabel, Hash, Clone, Debug, Eq, PartialEq)]
pub struct VoxelRender;

impl Plugin for NEVRPlugin {
    fn build(&self, app: &mut App) {
        let app = app.init_schedule(VoxelRender);

        app.world_mut()
            .resource_mut::<MainScheduleOrder>()
            .insert_before(PostUpdate, VoxelRender);

        app.insert_resource(
            VulkanInstance::new(Some(self.name.clone()), self.version.clone()).unwrap(),
        )
        .insert_resource(VoxelLight {
            ambient: Vec4::new(0.03, 0.03, 0.03, 1.0),
            direction: Vec4::NEG_Y,
            sky_color: Vec3::new(0.5, 0.7, 1.0),
        })
        .init_resource::<VoxelLibrary>()
        .init_resource::<VoxelWorld>()
        .init_resource::<DescriptorSets>()
        .add_systems(Startup, (init_images, setup))
        .add_systems(
            Update,
            (
                (init_camera, update_camera).chain(),
                update_camera_view,
                update_blocks,
                update_descriptor_set,
                update_sky,
            ),
        )
        .add_systems(VoxelRender, render);
    }
}

fn render(
    vulkan_instance: Res<VulkanInstance>,
    descriptor_sets: Res<DescriptorSets>,
    render_target: Res<VoxelRenderTarget>,
    mut images: ResMut<Assets<bevy::image::Image>>,
) {
    let image = images.get_mut(&render_target.0).unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        vulkan_instance.command_buffer_allocator(),
        vulkan_instance.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let descriptor_set = match descriptor_sets.descriptor_set.clone() {
        Some(d) => d,
        None => return,
    };

    let image_descriptor_sets = match descriptor_sets.image_descriptor_sets[0].clone() {
        Some(d) => d,
        None => return,
    };

    let intersect_descriptor_set = match descriptor_sets.intersect_descriptor_set.clone() {
        Some(d) => d,
        None => return,
    };

    let sky_color_descriptor_set = match descriptor_sets.sky_color_descriptor_set.clone() {
        Some(d) => d,
        None => return,
    };

    builder
        .bind_descriptor_sets(
            PipelineBindPoint::RayTracing,
            vulkan_instance.pipeline_layout(),
            0,
            vec![
                descriptor_set,
                image_descriptor_sets.1.clone(),
                intersect_descriptor_set,
                sky_color_descriptor_set,
            ],
        )
        .unwrap()
        .bind_pipeline_ray_tracing(vulkan_instance.pipeline())
        .unwrap();

    let extent = image_descriptor_sets.0.image().extent();

    unsafe { builder.trace_rays(vulkan_instance.shader_binding_table_addresses(), extent) }
        .unwrap();

    let buffer: Subbuffer<[u8]> = Buffer::new_unsized(
        vulkan_instance.memory_allocator(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        image.width() as DeviceSize * image.height() as DeviceSize * 4 * 4,
    )
    .unwrap();

    builder
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image_descriptor_sets.0.image().clone(),
            buffer.clone(),
        ))
        .unwrap();

    println!("building command buffer");
    let command_buffer = builder.build().unwrap();
    println!("executing and waiting command buffer");
    command_buffer
        .execute(vulkan_instance.queue())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    image.data = Some(buffer.read().unwrap().to_vec());
}

fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    let pixel = (0..4).map(|_| 1.0f32.to_le_bytes()).flatten().collect_vec();
    let image = bevy::image::Image::new_fill(
        Extent3d {
            width: 1920,
            height: 1080,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &pixel,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::default(),
    );

    let image = assets.add(image);

    commands.spawn(Camera2d::default());
    commands.spawn(Sprite::from_image(image.clone()));
    commands.insert_resource(VoxelRenderTarget(image));
}

fn init_images(vulkan_instance: Res<VulkanInstance>, mut descriptor_sets: ResMut<DescriptorSets>) {
    descriptor_sets.image_descriptor_sets = (0..3)
        .map(|_| {
            let image = Image::new(
                vulkan_instance.memory_allocator(),
                ImageCreateInfo {
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    extent: [1920, 1080, 1],
                    format: Format::R32G32B32A32_SFLOAT,
                    view_formats: vec![Format::R32G32B32A32_SFLOAT],
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap();

            let view = ImageView::new_default(image).unwrap();

            Some((
                view.clone(),
                DescriptorSet::new(
                    vulkan_instance.descriptor_set_allocator(),
                    vulkan_instance.pipeline_layout().set_layouts()[1].clone(),
                    [WriteDescriptorSet::image_view(0, view)],
                    [],
                )
                .unwrap(),
            ))
        })
        .collect_array::<3>()
        .unwrap();
}

fn init_camera(
    mut commands: Commands,
    camera_query: Query<(Entity, Ref<VoxelCamera>, Ref<GlobalTransform>)>,
    vulkan_instance: Res<VulkanInstance>,
) {
    for (entity, camera, transform) in camera_query {
        if !camera.is_added() || !transform.is_added() {
            continue;
        }
        commands
            .entity(entity)
            .insert(VoxelCameraData::new(&camera, &transform, &vulkan_instance).unwrap());
    }
}

fn update_camera(
    camera_query: Query<
        (&mut VoxelCameraData, &VoxelCamera, &GlobalTransform),
        Or<(Changed<VoxelCamera>, Changed<GlobalTransform>)>,
    >,
    vulkan_instance: Res<VulkanInstance>,
) {
    for (mut camera_data, camera, transform) in camera_query {
        camera_data
            .update_buffer(&vulkan_instance, camera, transform)
            .unwrap();
    }
}

fn update_camera_view(
    camera_query: Query<&mut VoxelCamera>,
    mut resize_reader: EventReader<WindowResized>,
    render_target: Res<VoxelRenderTarget>,
    vulkan_instance: Res<VulkanInstance>,
    mut images: ResMut<Assets<bevy::image::Image>>,
    mut descriptor_sets: ResMut<DescriptorSets>,
) {
    let mut width = 16.0;
    let mut height = 9.0;
    let mut changed = false;

    for event in resize_reader.read() {
        changed = true;
        width = event.width;
        height = event.height;
    }

    if !changed {
        return;
    }

    for mut camera in camera_query {
        camera.update_aspect_ratio(width, height);
    }
    images.get_mut(&render_target.0).unwrap().resize(Extent3d {
        width: width as u32,
        height: height as u32,
        depth_or_array_layers: 1,
    });

    descriptor_sets.image_descriptor_sets = (0..3)
        .map(|_| {
            let image = Image::new(
                vulkan_instance.memory_allocator(),
                ImageCreateInfo {
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    extent: [width as u32, height as u32, 1],
                    format: Format::R32G32B32A32_SFLOAT,
                    view_formats: vec![Format::R32G32B32A32_SFLOAT],
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap();

            let view = ImageView::new_default(image).unwrap();

            Some((
                view.clone(),
                DescriptorSet::new(
                    vulkan_instance.descriptor_set_allocator(),
                    vulkan_instance.pipeline_layout().set_layouts()[1].clone(),
                    [WriteDescriptorSet::image_view(0, view)],
                    [],
                )
                .unwrap(),
            ))
        })
        .collect_array::<3>()
        .unwrap();
}

fn update_blocks(
    blocks_query: Query<(Ref<VoxelBlock>, Ref<GlobalTransform>)>,
    library: Res<VoxelLibrary>,
    vulkan_instance: Res<VulkanInstance>,
    mut world: ResMut<VoxelWorld>,
    mut descriptor_sets: ResMut<DescriptorSets>,
) {
    // simple change detection
    let mut changed = false;
    for (block, transform) in blocks_query {
        if block.is_changed() || transform.is_changed() {
            changed = true;
            break;
        }
    }
    if !changed {
        return;
    }

    let (material_data, voxel_data) = world.update(blocks_query, &library, &vulkan_instance);
    descriptor_sets.intersect_descriptor_set = Some(
        DescriptorSet::new(
            vulkan_instance.descriptor_set_allocator(),
            vulkan_instance.pipeline_layout().set_layouts()[2].clone(),
            [
                WriteDescriptorSet::buffer(0, voxel_data),
                WriteDescriptorSet::buffer(1, material_data),
            ],
            [],
        )
        .unwrap(),
    );
}

fn update_descriptor_set(
    camera_query: Query<Ref<VoxelCameraData>>,
    world: Res<VoxelWorld>,
    vulkan_instance: Res<VulkanInstance>,
    mut descriptor_sets: ResMut<DescriptorSets>,
) {
    let camera_data = match camera_query.single() {
        Ok(c) => c,
        Err(_) => {
            println!("no camera");
            return;
        }
    };

    if camera_data.is_changed() || world.is_changed() {
        println!("updated descriptor");
        descriptor_sets.descriptor_set = Some(
            DescriptorSet::new(
                vulkan_instance.descriptor_set_allocator(),
                vulkan_instance.pipeline_layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::acceleration_structure(0, world.tlas()),
                    WriteDescriptorSet::buffer(1, camera_data.camera_gpu_buffer()),
                ],
                [],
            )
            .unwrap(),
        );
    }
}

fn update_sky(
    light: Res<VoxelLight>,
    vulkan_instance: Res<VulkanInstance>,
    mut descriptor_sets: ResMut<DescriptorSets>,
) {
    if !light.is_changed() {
        return;
    }

    println!("updated sky");

    let sky_color_buffer = Buffer::from_data(
        vulkan_instance.memory_allocator(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        [light.sky_color.x, light.sky_color.y, light.sky_color.z],
    )
    .unwrap();

    let light_buffer = Buffer::from_data(
        vulkan_instance.memory_allocator(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        Light {
            ambient_light: light.ambient.to_array(),
            light_direction: light.direction.to_array(),
        },
    )
    .unwrap();

    descriptor_sets.sky_color_descriptor_set = Some(
        DescriptorSet::new(
            vulkan_instance.descriptor_set_allocator(),
            vulkan_instance.pipeline_layout().set_layouts()[3].clone(),
            [
                WriteDescriptorSet::buffer(0, sky_color_buffer),
                WriteDescriptorSet::buffer(1, light_buffer),
            ],
            [],
        )
        .unwrap(),
    );
}
