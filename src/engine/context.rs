use crate::vulkan_instance::VulkanInstance;
use egui_winit_vulkano::{Gui, GuiConfig};
use std::iter;
use std::sync::Arc;
use vulkano::acceleration_structure::{
    AabbPositions, AccelerationStructure, AccelerationStructureBuildGeometryInfo,
    AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
    AccelerationStructureCreateInfo, AccelerationStructureGeometries,
    AccelerationStructureGeometryAabbsData, AccelerationStructureGeometryInstancesData,
    AccelerationStructureGeometryInstancesDataType, AccelerationStructureGeometryTrianglesData,
    AccelerationStructureInstance, AccelerationStructureType, BuildAccelerationStructureFlags,
    BuildAccelerationStructureMode,
};
use vulkano::buffer::{
    Buffer, BufferContents, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer,
};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
    PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageFormatInfo, ImageUsage, SampleCount};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::pipeline::ray_tracing::{
    RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
    ShaderBindingTable,
};
use vulkano::pipeline::{PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::ShaderStages;
use vulkano::swapchain::{PresentMode, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo};
use vulkano::sync;
use vulkano::sync::GpuFuture;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

#[derive(Debug, BufferContents, Copy, Clone)]
#[repr(C)]
pub struct Light {
    pub(crate) ambient_light: [f32; 4],
    pub(crate) light_direction: [f32; 4],
}

pub struct GraphicsContext {
    pub(crate) vulkan_instance: Arc<VulkanInstance>,
    pub(crate) window: Arc<Window>,
    pub(crate) swapchain: Arc<Swapchain>,
    pub(crate) previous_frame: Option<Box<dyn GpuFuture>>,
    pub(crate) recreate_swapchain: bool,
    pub(crate) swapchain_image_sets: Vec<(Arc<ImageView>, Arc<DescriptorSet>)>,
    pub(crate) pipeline_layout: Arc<PipelineLayout>,
    shader_binding_table: ShaderBindingTable,
    pipeline: Arc<RayTracingPipeline>,
    pub(crate) builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    pub(crate) image_index: Option<u32>,
    pub(crate) gui: Gui,
}

impl GraphicsContext {
    pub fn new(
        vulkan_instance: Arc<VulkanInstance>,
        event_loop: &ActiveEventLoop,
        attributes: WindowAttributes,
    ) -> Option<Self> {
        let window = Arc::new(event_loop.create_window(attributes).ok()?);

        let surface = Surface::from_window(vulkan_instance.instance(), window.clone()).ok()?;
        let window_size = window.inner_size();

        let surface_capabilities = vulkan_instance
            .device()
            .physical_device()
            .surface_capabilities(&surface, SurfaceInfo::default())
            .ok()?;

        let (image_format, image_color_space) = vulkan_instance
            .device()
            .physical_device()
            .surface_formats(&surface, SurfaceInfo::default())
            .ok()?
            .into_iter()
            .find(|(format, _)| {
                vulkan_instance
                    .device()
                    .physical_device()
                    .image_format_properties(ImageFormatInfo {
                        format: *format,
                        usage: ImageUsage::STORAGE | ImageUsage::COLOR_ATTACHMENT,
                        ..Default::default()
                    })
                    .unwrap()
                    .is_some()
            })?;

        let gui = Gui::new(
            event_loop,
            surface.clone(),
            vulkan_instance.queue(),
            image_format,
            GuiConfig {
                samples: SampleCount::Sample1,
                allow_srgb_render_target: true,
                is_overlay: true,
            },
        );

        let (swapchain, images) = Swapchain::new(
            vulkan_instance.device(),
            surface,
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_color_space,
                image_extent: window_size.into(),
                image_usage: ImageUsage::STORAGE | ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()?,
                present_mode: PresentMode::Immediate,
                ..Default::default()
            },
        )
        .ok()?;

        let pipeline_layout = PipelineLayout::new(
            vulkan_instance.device(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![
                    DescriptorSetLayout::new(
                        vulkan_instance.device(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: [
                                (
                                    0,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::RAYGEN | ShaderStages::CLOSEST_HIT,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::AccelerationStructure,
                                        )
                                    },
                                ),
                                (
                                    1,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::RAYGEN,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::UniformBuffer,
                                        )
                                    },
                                ),
                            ]
                            .into(),
                            ..Default::default()
                        },
                    )
                    .ok()?,
                    DescriptorSetLayout::new(
                        vulkan_instance.device(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: [(
                                0,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::RAYGEN,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::StorageImage,
                                    )
                                },
                            )]
                            .into(),
                            ..Default::default()
                        },
                    )
                    .ok()?,
                    DescriptorSetLayout::new(
                        vulkan_instance.device(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: [
                                (
                                    0,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::INTERSECTION
                                            | ShaderStages::CLOSEST_HIT,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::StorageBuffer,
                                        )
                                    },
                                ),
                                (
                                    1,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::CLOSEST_HIT,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::StorageBuffer,
                                        )
                                    },
                                ),
                            ]
                            .into(),
                            ..Default::default()
                        },
                    )
                    .ok()?,
                    DescriptorSetLayout::new(
                        vulkan_instance.device(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: [
                                (
                                    0,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::MISS,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::StorageBuffer,
                                        )
                                    },
                                ),
                                (
                                    1,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::RAYGEN | ShaderStages::CLOSEST_HIT,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::UniformBuffer,
                                        )
                                    },
                                ),
                            ]
                            .into(),
                            ..Default::default()
                        },
                    )
                    .ok()?,
                ],
                ..Default::default()
            },
        )
        .ok()?;

        let pipeline = {
            let raygen = raygen::load(vulkan_instance.device())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let closest_hit = raychit::load(vulkan_instance.device())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let miss = raymiss::load(vulkan_instance.device())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let intersect = rayintersect::load(vulkan_instance.device())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let shadow = rayshadow::load(vulkan_instance.device())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(raygen),
                PipelineShaderStageCreateInfo::new(closest_hit),
                PipelineShaderStageCreateInfo::new(miss),
                PipelineShaderStageCreateInfo::new(intersect),
                PipelineShaderStageCreateInfo::new(shadow),
            ];

            let groups = [
                RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
                RayTracingShaderGroupCreateInfo::General { general_shader: 2 },
                RayTracingShaderGroupCreateInfo::General { general_shader: 4 },
                RayTracingShaderGroupCreateInfo::ProceduralHit {
                    closest_hit_shader: Some(1),
                    any_hit_shader: None,
                    intersection_shader: 3,
                },
            ];

            RayTracingPipeline::new(
                vulkan_instance.device(),
                None,
                RayTracingPipelineCreateInfo {
                    stages: stages.to_vec().into(),
                    groups: groups.to_vec().into(),
                    max_pipeline_ray_recursion_depth: 2,
                    ..RayTracingPipelineCreateInfo::layout(pipeline_layout.clone())
                },
            )
            .ok()?
        };

        let swapchain_image_sets = window_size_dependent_setup(
            images,
            pipeline_layout.clone(),
            vulkan_instance.descriptor_set_allocator(),
        );

        let shader_binding_table =
            ShaderBindingTable::new(vulkan_instance.memory_allocator(), &pipeline).unwrap();

        let previous_frame = Some(sync::now(vulkan_instance.device()).boxed());

        Some(Self {
            vulkan_instance,
            window,
            swapchain,
            previous_frame,
            swapchain_image_sets,
            pipeline_layout,
            shader_binding_table,
            pipeline,
            gui,
            recreate_swapchain: false,
            builder: None,
            image_index: None,
        })
    }

    pub(crate) fn resize(&mut self, images: Vec<Arc<Image>>) {
        self.swapchain_image_sets = window_size_dependent_setup(
            images,
            self.pipeline_layout.clone(),
            self.vulkan_instance.descriptor_set_allocator(),
        );
    }

    pub(crate) fn draw(
        &mut self,
        descriptor_set: Arc<DescriptorSet>,
        intersect_descriptor_set: Arc<DescriptorSet>,
        sky_color_descriptor_set: Arc<DescriptorSet>,
    ) {
        let builder = self.builder.as_mut().unwrap();

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::RayTracing,
                self.pipeline_layout.clone(),
                0,
                vec![
                    descriptor_set,
                    self.swapchain_image_sets[self.image_index.unwrap() as usize]
                        .1
                        .clone(),
                    intersect_descriptor_set,
                    sky_color_descriptor_set,
                ],
            )
            .unwrap()
            .bind_pipeline_ray_tracing(self.pipeline.clone())
            .unwrap();

        let extent = self.swapchain_image_sets[self.image_index.unwrap() as usize]
            .0
            .image()
            .extent();

        unsafe { builder.trace_rays(self.shader_binding_table.addresses().clone(), extent) }
            .unwrap();
    }
}

mod raygen {
    vulkano_shaders::shader! {
        ty: "raygen",
        path: "./shaders/rgen.glsl",
        vulkan_version: "1.3"
    }
}

mod raychit {
    vulkano_shaders::shader! {
        ty: "closesthit",
        path: "./shaders/rchit.glsl",
        vulkan_version: "1.3"
    }
}

mod raymiss {
    vulkano_shaders::shader! {
        ty: "miss",
        path: "./shaders/rmiss.glsl",
        vulkan_version: "1.3"
    }
}

mod rayintersect {
    vulkano_shaders::shader! {
        ty: "intersection",
        path: "./shaders/rintersect.glsl",
        vulkan_version: "1.3"
    }
}

mod rayshadow {
    vulkano_shaders::shader! {
        ty: "miss",
        path: "./shaders/rmiss_shadow.glsl",
        vulkan_version: "1.3"
    }
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: Vec<Arc<Image>>,
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
) -> Vec<(Arc<ImageView>, Arc<DescriptorSet>)> {
    let swapchain_image_sets = images
        .into_iter()
        .map(|image| {
            let image_view = ImageView::new_default(image).unwrap();
            let descriptor_set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                pipeline_layout.set_layouts()[1].clone(),
                [WriteDescriptorSet::image_view(0, image_view.clone())],
                [],
            )
            .unwrap();

            (image_view, descriptor_set)
        })
        .collect();

    swapchain_image_sets
}

/// A helper function to build an acceleration structure and wait for its completion.
///
/// # Safety
///
/// - If you are referencing a bottom-level acceleration structure in a top-level acceleration
///   structure, you must ensure that the bottom-level acceleration structure is kept alive.
unsafe fn build_acceleration_structure_common(
    geometries: AccelerationStructureGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: BuildAccelerationStructureMode::Build,
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let as_build_sizes_info = device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &as_build_geometry_info,
            &[primitive_count],
        )
        .unwrap();

    // We create a new scratch buffer for each acceleration structure for simplicity. You may want
    // to reuse scratch buffers if you need to build many acceleration structures.
    let scratch_buffer = Buffer::new_slice::<u8>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
        as_build_sizes_info.build_scratch_size,
    )
    .unwrap();

    let acceleration = unsafe {
        AccelerationStructure::new(
            device,
            AccelerationStructureCreateInfo {
                ty,
                ..AccelerationStructureCreateInfo::new(
                    Buffer::new_slice::<u8>(
                        memory_allocator,
                        BufferCreateInfo {
                            usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                                | BufferUsage::SHADER_DEVICE_ADDRESS,
                            ..Default::default()
                        },
                        AllocationCreateInfo::default(),
                        as_build_sizes_info.acceleration_structure_size,
                    )
                    .unwrap(),
                )
            },
        )
    }
    .unwrap();

    as_build_geometry_info.dst_acceleration_structure = Some(acceleration.clone());
    as_build_geometry_info.scratch_data = Some(scratch_buffer);

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
        ..Default::default()
    };

    // For simplicity, we build a single command buffer that builds the acceleration structure,
    // then waits for its execution to complete.
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    unsafe {
        builder.build_acceleration_structure(
            as_build_geometry_info,
            iter::once(as_build_range_info).collect(),
        )
    }
    .unwrap();

    builder
        .build()
        .unwrap()
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    acceleration
}

pub(crate) unsafe fn build_acceleration_structure_voxels(
    voxel_buffer: &Subbuffer<[AabbPositions]>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let primitive_count = voxel_buffer.len() as u32;
    let as_geometry_voxels_data = AccelerationStructureGeometryAabbsData {
        data: Some(voxel_buffer.clone().into_bytes()),
        stride: size_of::<AabbPositions>() as u32,
        ..AccelerationStructureGeometryAabbsData::default()
    };

    let geometries = AccelerationStructureGeometries::Aabbs(vec![as_geometry_voxels_data]);

    unsafe {
        build_acceleration_structure_common(
            geometries,
            primitive_count,
            AccelerationStructureType::BottomLevel,
            memory_allocator,
            command_buffer_allocator,
            device,
            queue,
        )
    }
}

pub(crate) unsafe fn build_acceleration_structure_triangles(
    primitive_count: u32,
    vertex_buffer: Subbuffer<[[i32; 3]]>,
    index_buffer: Subbuffer<[u32]>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let as_geometry_triangles_data = AccelerationStructureGeometryTrianglesData {
        max_vertex: vertex_buffer.len() as _,
        vertex_data: Some(vertex_buffer.clone().into_bytes()),
        vertex_stride: size_of::<[i32; 3]>() as _,
        index_data: Some(IndexBuffer::U32(index_buffer)),
        ..AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT)
    };

    let geometries = AccelerationStructureGeometries::Triangles(vec![as_geometry_triangles_data]);

    unsafe {
        build_acceleration_structure_common(
            geometries,
            primitive_count,
            AccelerationStructureType::BottomLevel,
            memory_allocator,
            command_buffer_allocator,
            device,
            queue,
        )
    }
}

pub(crate) unsafe fn build_top_level_acceleration_structure(
    as_instances: Vec<AccelerationStructureInstance>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let primitive_count = as_instances.len() as u32;

    let instance_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        as_instances,
    )
    .unwrap();

    let as_geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
        AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer)),
    );

    let geometries = AccelerationStructureGeometries::Instances(as_geometry_instances_data);

    unsafe {
        build_acceleration_structure_common(
            geometries,
            primitive_count,
            AccelerationStructureType::TopLevel,
            memory_allocator,
            command_buffer_allocator,
            device,
            queue,
        )
    }
}
