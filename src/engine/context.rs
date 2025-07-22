use crate::engine::vulkan::Vulkan;
use crate::engine::vulkan::surface::VulkanSurface;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::pipeline::VulkanPipeline;
use crate::vulkan::shader::VulkanShader;
use crate::vulkan::swapchain::VulkanSwapchain;
use crate::vulkan::{VulkanApplicationInfo, VulkanInstanceCreateInfo};
use ash::vk;
use ash::vk::{DescriptorType, MemoryAllocateFlags, ShaderStageFlags};
use std::ffi::CStr;
use std::ptr::null;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

pub struct GraphicsContext {
    window: Window,
    raygen_shader: VulkanShader,
    raychit_shader: VulkanShader,
    raymiss_shader: VulkanShader,
    raymiss_shadow_shader: VulkanShader,
    pipeline: VulkanPipeline,
    swapchain: VulkanSwapchain,
    device: VulkanDevice,
    surface: VulkanSurface,
    vulkan: Vulkan,
}

impl GraphicsContext {
    pub fn new(
        app_info: VulkanApplicationInfo,
        event_loop: &ActiveEventLoop,
        attributes: WindowAttributes,
    ) -> Option<Self> {
        let window = event_loop.create_window(attributes).ok()?;

        let vulkan = Vulkan::new(
            VulkanInstanceCreateInfo::default()
                .with_app_info(app_info)
                .with_extensions(vec!["VK_KHR_surface"])
                .enable_debug()
                .add_required_extensions(&window),
        )
        .unwrap();

        let surface = vulkan.create_surface(&window).ok()?;
        let (physical_device, queue_family_index) = vulkan.find_physical_device(&surface)?;

        let mut raytracing_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR {
            ..Default::default()
        };

        let properties =
            vulkan.add_raytracing_properties(&physical_device, &mut raytracing_properties);
        let memory_properties = vulkan.physical_device_memory_properties(&physical_device);

        unsafe {
            println!(
                "Using device {}",
                CStr::from_ptr(properties.properties.device_name.as_ptr())
                    .to_str()
                    .unwrap()
            );
        }

        let mut address_features = vk::PhysicalDeviceBufferDeviceAddressFeatures {
            buffer_device_address: true.into(),
            buffer_device_address_capture_replay: false.into(),
            buffer_device_address_multi_device: false.into(),
            ..Default::default()
        };

        let pointer_address_features: *mut vk::PhysicalDeviceBufferDeviceAddressFeatures =
            &mut address_features;

        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
                p_next: pointer_address_features.cast(),
                acceleration_structure: true.into(),
                acceleration_structure_capture_replay: false.into(),
                acceleration_structure_indirect_build: false.into(),
                acceleration_structure_host_commands: false.into(),
                descriptor_binding_acceleration_structure_update_after_bind: false.into(),
                ..Default::default()
            };

        let pointer_acceleration_structure_features:
            *mut vk::PhysicalDeviceAccelerationStructureFeaturesKHR = &mut acceleration_structure_features;

        let raytracing_pipeline_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR {
            p_next: pointer_acceleration_structure_features.cast(),
            ray_tracing_pipeline: true.into(),
            ray_tracing_pipeline_shader_group_handle_capture_replay: false.into(),
            ray_tracing_pipeline_shader_group_handle_capture_replay_mixed: false.into(),
            ray_tracing_pipeline_trace_rays_indirect: false.into(),
            ray_traversal_primitive_culling: false.into(),
            ..Default::default()
        };

        let device_features = vk::PhysicalDeviceFeatures {
            geometry_shader: true.into(),
            ..Default::default()
        };

        let descriptor_pool_sizes = vec![
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 4,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
            },
        ];

        let descriptor_set_layout_bindings = vec![
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::RAYGEN_KHR | ShaderStageFlags::CLOSEST_HIT_KHR,
                p_immutable_samplers: null(),
                _marker: Default::default(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::RAYGEN_KHR | ShaderStageFlags::CLOSEST_HIT_KHR,
                p_immutable_samplers: null(),
                _marker: Default::default(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::CLOSEST_HIT_KHR,
                p_immutable_samplers: null(),
                _marker: Default::default(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::CLOSEST_HIT_KHR,
                p_immutable_samplers: null(),
                _marker: Default::default(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 4,
                descriptor_type: DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::RAYGEN_KHR,
                p_immutable_samplers: null(),
                _marker: Default::default(),
            },
        ];

        let material_descriptor_set_layout_bindings = vec![
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::CLOSEST_HIT_KHR,
                p_immutable_samplers: null(),
                _marker: Default::default(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::CLOSEST_HIT_KHR,
                p_immutable_samplers: null(),
                _marker: Default::default(),
            },
        ];

        let mut device = VulkanDevice::new(
            &vulkan,
            &physical_device,
            queue_family_index,
            [
                "VK_KHR_ray_tracing_pipeline",
                "VK_KHR_acceleration_structure",
                "VK_EXT_descriptor_indexing",
                "VK_KHR_maintenance3",
                "VK_KHR_buffer_device_address",
                "VK_KHR_deferred_host_operations",
                "VK_KHR_swapchain",
            ],
            &raytracing_pipeline_features,
            &device_features,
            descriptor_pool_sizes,
            descriptor_set_layout_bindings,
            material_descriptor_set_layout_bindings,
        )
        .unwrap();

        let memory_allocate_flags = vk::MemoryAllocateFlagsInfo {
            flags: MemoryAllocateFlags::DEVICE_ADDRESS,
            device_mask: 0,
            ..Default::default()
        };

        let swapchain =
            VulkanSwapchain::new(&vulkan, &physical_device, &mut device, &surface).unwrap();

        let raygen_shader = VulkanShader::new_with_content(
            include_bytes!("../../shaders/shader.rgen.spv"),
            &device,
        )
        .unwrap();
        let raychit_shader = VulkanShader::new_with_content(
            include_bytes!("../../shaders/shader.rchit.spv"),
            &device,
        )
        .unwrap();
        let raymiss_shader = VulkanShader::new_with_content(
            include_bytes!("../../shaders/shader.rmiss.spv"),
            &device,
        )
        .unwrap();
        let raymiss_shadow_shader = VulkanShader::new_with_content(
            include_bytes!("../../shaders/shader_shadow.rmiss.spv"),
            &device,
        )
        .unwrap();

        let pipeline = device
            .create_pipeline([
                &raygen_shader,
                &raychit_shader,
                &raymiss_shader,
                &raymiss_shadow_shader,
            ])
            .unwrap();

        Some(Self {
            vulkan,
            window,
            surface,
            device,
            swapchain,
            pipeline,
            raygen_shader,
            raychit_shader,
            raymiss_shader,
            raymiss_shadow_shader,
        })
    }
}
