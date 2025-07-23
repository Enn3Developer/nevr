use crate::buffer::VulkanBuffer;
use foldhash::HashMap;
use std::iter;
use std::sync::Arc;
use vulkano::acceleration_structure::{
    AccelerationStructure, AccelerationStructureBuildGeometryInfo,
    AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
    AccelerationStructureCreateInfo, AccelerationStructureGeometries,
    AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
    AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance,
    AccelerationStructureType, BuildAccelerationStructureFlags, BuildAccelerationStructureMode,
    TransformMatrix,
};
use vulkano::buffer::{
    Buffer, BufferContents, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer,
};
use vulkano::command_buffer::allocator::{
    CommandBufferAllocator, StandardCommandBufferAllocator,
    StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferLevel, CommandBufferSubmitInfo, CommandBufferUsage,
    CopyImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, SubmitInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::pool::{
    DescriptorPool, DescriptorPoolAlloc, DescriptorPoolCreateInfo, DescriptorSetAllocateInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::sampler::ComponentMapping;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{
    Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling,
    ImageType, ImageUsage, SampleCount,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::pipeline::ray_tracing::{
    RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
};
use vulkano::pipeline::{PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::ShaderStages;
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo};
use vulkano::sync::fence::{Fence, FenceCreateInfo};
use vulkano::sync::{GpuFuture, ImageMemoryBarrier};
use vulkano::{Version, VulkanLibrary};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

struct Material {
    ambient: [f32; 3],
    diffuse: [f32; 3],
    specular: [f32; 3],
    emission: [f32; 3],
}

#[derive(BufferContents, Clone)]
#[repr(C)]
struct Camera {
    camera_position: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    camera_forward: [f32; 4],

    frame_count: u32,
}

pub struct GraphicsContext {
    window: Arc<Window>,
    library: Arc<VulkanLibrary>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    surface: Arc<Surface>,
    swapchain: Arc<Swapchain>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pipeline: Arc<RayTracingPipeline>,
    blas: Arc<AccelerationStructure>,
    tlas: Arc<AccelerationStructure>,
    camera: Camera,
    camera_buffer: VulkanBuffer<Camera>,
}

impl GraphicsContext {
    pub fn new(
        app_name: &str,
        app_version: &Version,
        event_loop: &ActiveEventLoop,
        attributes: WindowAttributes,
    ) -> Option<Self> {
        let window = Arc::new(event_loop.create_window(attributes).ok()?);
        let required_extensions = Surface::required_extensions(&event_loop).unwrap();

        let library = VulkanLibrary::new().unwrap();
        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                application_version: *app_version,
                application_name: Some(app_name.to_string()),
                enabled_extensions: required_extensions,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .unwrap();

        let extensions = DeviceExtensions {
            khr_ray_tracing_pipeline: true,
            khr_acceleration_structure: true,
            ext_descriptor_indexing: true,
            khr_maintenance3: true,
            khr_buffer_device_address: true,
            khr_deferred_host_operations: true,
            khr_swapchain: true,
            ..Default::default()
        };

        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .unwrap();

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_features: DeviceFeatures {
                    acceleration_structure: true,
                    buffer_device_address: true,
                    ray_tracing_pipeline: true,
                    geometry_shader: true,
                    ..Default::default()
                },
                enabled_extensions: extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        let dimensions = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1,
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::TRANSFER_DST,
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap();

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
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
                            stages: ShaderStages::RAYGEN | ShaderStages::CLOSEST_HIT,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::UniformBuffer,
                            )
                        },
                    ),
                    (
                        2,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::CLOSEST_HIT,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                    (
                        3,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::CLOSEST_HIT,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                    (
                        4,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::RAYGEN,
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageImage,
                            )
                        },
                    ),
                ]
                .into(),
                ..Default::default()
            },
        )
        .unwrap();

        let material_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: [
                    (
                        0,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages::CLOSEST_HIT,
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
        .unwrap();

        let descriptor_set_layouts = vec![
            descriptor_set_layout.clone(),
            material_descriptor_set_layout.clone(),
        ];

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: descriptor_set_layouts,
                ..Default::default()
            },
        )
        .unwrap();

        let rchit = raychit::load(device.clone()).unwrap();
        let raygen = raygen::load(device.clone()).unwrap();
        let rmiss = raymiss::load(device.clone()).unwrap();
        let rmiss_shadow = raymiss_shadow::load(device.clone()).unwrap();

        let pipeline = RayTracingPipeline::new(
            device.clone(),
            None,
            RayTracingPipelineCreateInfo {
                stages: vec![
                    PipelineShaderStageCreateInfo::new(rchit.entry_point("main").unwrap()),
                    PipelineShaderStageCreateInfo::new(raygen.entry_point("main").unwrap()),
                    PipelineShaderStageCreateInfo::new(rmiss.entry_point("main").unwrap()),
                    PipelineShaderStageCreateInfo::new(rmiss_shadow.entry_point("main").unwrap()),
                ]
                .into(),
                groups: vec![
                    RayTracingShaderGroupCreateInfo::TrianglesHit {
                        closest_hit_shader: Some(0),
                        any_hit_shader: None,
                    },
                    RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
                    RayTracingShaderGroupCreateInfo::General { general_shader: 2 },
                    RayTracingShaderGroupCreateInfo::General { general_shader: 3 },
                ]
                .into(),
                max_pipeline_ray_recursion_depth: 1,
                ..RayTracingPipelineCreateInfo::layout(pipeline_layout)
            },
        )
        .unwrap();

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count: 16,
                ..Default::default()
            },
        ));

        let vertices = [[10, 10, 10], [10, 11, 10], [11, 10, 10], [10, 11, 10]];
        let indices = [0, 1, 2, 1, 2, 3];
        let primitive_count = 2;
        let materials = [
            Material {
                ambient: [0.0, 0.0, 0.0],
                diffuse: [1.0, 0.0, 0.0],
                specular: [0.2, 0.2, 0.2],
                emission: [0.5, 0.0, 0.0],
            },
            Material {
                ambient: [0.0, 0.0, 0.0],
                diffuse: [0.0, 0.0, 1.0],
                specular: [0.8, 0.8, 0.8],
                emission: [0.0, 0.0, 0.5],
            },
        ];

        let vertex_buffer = VulkanBuffer::new(
            memory_allocator.clone(),
            vertices,
            BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::STORAGE_BUFFER,
        );
        let index_buffer = VulkanBuffer::new(
            memory_allocator.clone(),
            indices,
            BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::STORAGE_BUFFER,
        );

        let blas = unsafe {
            build_acceleration_structure_triangles(
                primitive_count,
                vertex_buffer.device_buffer.clone(),
                index_buffer.device_buffer.clone(),
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                device.clone(),
                queue.clone(),
            )
        };

        let tlas = unsafe {
            build_top_level_acceleration_structure(
                vec![AccelerationStructureInstance {
                    acceleration_structure_reference: blas.device_address().into(),
                    transform: TransformMatrix::from([
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]),
                    ..Default::default()
                }],
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                device.clone(),
                queue.clone(),
            )
        };

        let camera = Camera {
            camera_position: [0.0, 0.0, 0.0, 1.0],
            camera_right: [1.0, 0.0, 0.0, 1.0],
            camera_up: [0.0, 1.0, 0.0, 1.0],
            camera_forward: [0.0, 0.0, 1.0, 1.0],
            frame_count: 0,
        };

        let camera_buffer = VulkanBuffer::new_with_data(
            memory_allocator.clone(),
            camera.clone(),
            BufferUsage::UNIFORM_BUFFER,
        );

        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: image_format,
                extent: [dimensions.width, dimensions.height, 1],
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCount::Sample1,
                tiling: ImageTiling::Optimal,
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        let image_view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Dim2d,
                format: image_format,
                component_mapping: ComponentMapping::identity(),
                subresource_range: ImageSubresourceRange {
                    aspects: ImageAspects::COLOR,
                    mip_levels: 0..1,
                    array_layers: 0..1,
                },
                ..Default::default()
            },
        )
        .unwrap();

        let image_memory_barrier = ImageMemoryBarrier {
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::General,
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects::COLOR,
                mip_levels: 0..1,
                array_layers: 0..1,
            },
            ..ImageMemoryBarrier::image(image.clone())
        };

        let image_fence = Fence::new(device.clone(), FenceCreateInfo::default()).unwrap();

        let builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let fence = builder
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        fence.wait(None).unwrap();

        let write_descriptor_sets = vec![
            WriteDescriptorSet::acceleration_structure(0, tlas.clone()),
            WriteDescriptorSet::buffer(1, camera_buffer.device_buffer.clone()),
            WriteDescriptorSet::buffer(2, index_buffer.device_buffer.clone()),
            WriteDescriptorSet::buffer(3, vertex_buffer.device_buffer.clone()),
            WriteDescriptorSet::image_view(4, image_view.clone()),
        ];

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            descriptor_set_layout,
            write_descriptor_sets,
            None,
        )
        .unwrap();

        Some(Self {
            window,
            library,
            instance,
            device,
            surface,
            swapchain,
            queue,
            memory_allocator,
            command_buffer_allocator,
            pipeline,
            blas,
            tlas,
            camera,
            camera_buffer,
        })
    }
}

mod raygen {
    vulkano_shaders::shader! {
        ty: "raygen",
        path: "./shaders/shader.rgen",
        vulkan_version: "1.3"
    }
}

mod raychit {
    vulkano_shaders::shader! {
        ty: "closesthit",
        path: "./shaders/shader.rchit",
        vulkan_version: "1.3"
    }
}

mod raymiss {
    vulkano_shaders::shader! {
        ty: "miss",
        path: "./shaders/shader.rmiss",
        vulkan_version: "1.3"
    }
}

mod raymiss_shadow {
    vulkano_shaders::shader! {
        ty: "miss",
        path: "./shaders/shader_shadow.rmiss",
        vulkan_version: "1.3"
    }
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
    memory_allocator: Arc<StandardMemoryAllocator>,
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

unsafe fn build_acceleration_structure_triangles(
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

unsafe fn build_top_level_acceleration_structure(
    as_instances: Vec<AccelerationStructureInstance>,
    memory_allocator: Arc<StandardMemoryAllocator>,
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
