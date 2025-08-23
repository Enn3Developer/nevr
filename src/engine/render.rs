use crate::engine::DeviceSize;
use crate::engine::vulkan_instance::VulkanInstance;
use crate::{DescriptorSets, VoxelRenderTarget};
use bevy::app::{App, MainScheduleOrder, PostUpdate};
use bevy::asset::Assets;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::{
    Commands, EventReader, NonSendMut, Plugin, Res, ResMut, Resource, Startup, Update,
};
use bevy::window::WindowResized;
use itertools::Itertools;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
    PrimaryCommandBufferAbstract,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::sync::GpuFuture;

pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

#[derive(ScheduleLabel, Hash, Clone, Debug, Eq, PartialEq)]
pub struct VoxelRender;

#[derive(Resource, Default)]
pub struct FramesInFlight(usize);

impl FramesInFlight {
    pub fn get(&self) -> usize {
        self.0
    }

    pub fn inc(&mut self) {
        self.0 += 1;
        if self.0 >= MAX_FRAMES_IN_FLIGHT {
            self.0 = 0;
        }
    }

    pub fn get_and_inc(&mut self) -> usize {
        let frame = self.get();
        self.inc();

        frame
    }
}

#[derive(Default)]
pub struct FrameFutures {
    pub futures: [Option<Box<dyn GpuFuture>>; MAX_FRAMES_IN_FLIGHT],
}

#[derive(Resource)]
pub struct BufferedImages {
    pub buffers: [Subbuffer<[u8]>; MAX_FRAMES_IN_FLIGHT],
}

impl BufferedImages {
    pub fn new(vulkan_instance: &VulkanInstance, width: DeviceSize, height: DeviceSize) -> Self {
        Self {
            buffers: Self::create_buffers(vulkan_instance, width, height),
        }
    }

    pub fn resize(
        &mut self,
        vulkan_instance: &VulkanInstance,
        width: DeviceSize,
        height: DeviceSize,
    ) {
        self.buffers = Self::create_buffers(vulkan_instance, width, height);
    }

    fn create_buffers(
        vulkan_instance: &VulkanInstance,
        width: DeviceSize,
        height: DeviceSize,
    ) -> [Subbuffer<[u8]>; MAX_FRAMES_IN_FLIGHT] {
        const BYTES_PER_CHANNEL: DeviceSize = 4;
        const CHANNELS: DeviceSize = 4;

        (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Buffer::new_unsized(
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
                    width * height * BYTES_PER_CHANNEL * CHANNELS,
                )
                .unwrap()
            })
            .collect_array()
            .unwrap()
    }
}

pub struct VoxelRenderPlugin;

impl Plugin for VoxelRenderPlugin {
    fn build(&self, app: &mut App) {
        let app = app.init_schedule(VoxelRender);

        app.world_mut()
            .resource_mut::<MainScheduleOrder>()
            .insert_before(PostUpdate, VoxelRender);

        app.init_resource::<FramesInFlight>()
            .init_non_send_resource::<FrameFutures>()
            .add_systems(Startup, setup)
            .add_systems(Update, resize)
            .add_systems(VoxelRender, render);
    }
}

fn setup(mut commands: Commands, vulkan_instance: Res<VulkanInstance>) {
    commands.insert_resource(BufferedImages::new(&vulkan_instance, 1920, 1080));
}

fn resize(
    vulkan_instance: Res<VulkanInstance>,
    mut buffered_images: ResMut<BufferedImages>,
    mut resized_events: EventReader<WindowResized>,
) {
    let mut width: DeviceSize = 1920;
    let mut height: DeviceSize = 1080;
    let mut changed = false;

    for event in resized_events.read() {
        changed = true;
        width = event.width as DeviceSize;
        height = event.height as DeviceSize;
    }

    if !changed {
        return;
    }

    buffered_images.resize(&vulkan_instance, width, height);
}

fn render(
    vulkan_instance: Res<VulkanInstance>,
    descriptor_sets: Res<DescriptorSets>,
    render_target: Res<VoxelRenderTarget>,
    buffered_images: Res<BufferedImages>,
    mut images: ResMut<Assets<bevy::image::Image>>,
    mut frames_in_flight: ResMut<FramesInFlight>,
    mut frame_futures: NonSendMut<FrameFutures>,
) {
    let frame_in_flight = frames_in_flight.get_and_inc();
    let mut frame = match frame_futures.futures[frame_in_flight].take() {
        Some(f) => f,
        None => vulkano::sync::now(vulkan_instance.device()).boxed(),
    };

    frame.cleanup_finished();

    let image = images.get_mut(&render_target.0).unwrap();
    let buffer = buffered_images.buffers[frame_in_flight].clone();
    image.data = Some(
        match buffer.read() {
            Ok(b) => b,
            Err(_) => {
                // wait for frame to finish
                frame
                    .join(vulkano::sync::now(vulkan_instance.device()))
                    .then_signal_fence()
                    .wait(None)
                    .unwrap();

                buffer.read().unwrap()
            }
        }
        .to_vec(),
    );

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

    let image_descriptor_sets = match descriptor_sets.image_descriptor_sets[frame_in_flight].clone()
    {
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

    builder
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image_descriptor_sets.0.image().clone(),
            buffer.clone(),
        ))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = command_buffer
        .execute(vulkan_instance.queue())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    frame_futures.futures[frame_in_flight] = Some(future.boxed());
}
