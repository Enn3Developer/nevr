use crate::engine::vulkan::Vulkan;
use crate::engine::vulkan::surface::VulkanSurface;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::{VulkanApplicationInfo, VulkanInstanceCreateInfo};
use ash::vk;
use std::ffi::CStr;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

pub struct GraphicsContext {
    vulkan: Vulkan,
    window: Window,
    surface: VulkanSurface,
    device: VulkanDevice,
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

        let device = VulkanDevice::new(
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
        )
        .unwrap();

        Some(Self {
            vulkan,
            window,
            surface,
            device,
        })
    }
}
