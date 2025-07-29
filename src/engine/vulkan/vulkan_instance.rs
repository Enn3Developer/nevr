use std::sync::Arc;
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::{
    FreeListAllocator, GenericMemoryAllocator, GenericMemoryAllocatorCreateInfo,
    StandardMemoryAllocator,
};
use vulkano::{Validated, Version, VulkanLibrary};
use winit::raw_window_handle::HandleError;

pub struct VulkanInstance {
    instance: Arc<Instance>,
    queue_family_index: u32,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl VulkanInstance {
    pub fn new(application_name: Option<String>, application_version: Version) -> Option<Self> {
        let vulkan = VulkanLibrary::new().ok()?;

        #[cfg(target_os = "linux")]
        let required_extensions = InstanceExtensions {
            khr_wayland_surface: true,
            khr_xcb_surface: true,
            khr_xlib_surface: true,
            ..InstanceExtensions::empty()
        };

        #[cfg(target_os = "windows")]
        let required_extensions = InstanceExtensions {
            khr_win32_surface: true,
            ..InstanceExtensions::empty()
        };

        #[cfg(target_os = "macos")]
        let required_extensions = InstanceExtensions {
            ext_metal_surface: true,
            ..InstanceExtensions::empty()
        };

        #[cfg(target_os = "android")]
        let required_extensions = InstanceExtensions {
            khr_android_surface: true,
            ..InstanceExtensions::empty()
        };

        let instance = Instance::new(
            vulkan,
            InstanceCreateInfo {
                application_name,
                application_version,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: InstanceExtensions {
                    khr_surface: true,
                    ext_swapchain_colorspace: true,
                    ..required_extensions
                },
                ..Default::default()
            },
        )
        .ok()?;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_ray_tracing_pipeline: true,
            khr_ray_tracing_maintenance1: true,
            khr_synchronization2: true,
            khr_deferred_host_operations: true,
            khr_acceleration_structure: true,
            khr_push_descriptor: true,
            ..DeviceExtensions::empty()
        };

        let device_features = DeviceFeatures {
            acceleration_structure: true,
            ray_tracing_pipeline: true,
            buffer_device_address: true,
            synchronization2: true,
            ..DeviceFeatures::default()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.api_version() >= Version::V1_3)
            .filter(|p| {
                p.supported_extensions().contains(&device_extensions)
                    && p.supported_features().contains(&device_features)
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags
                            .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                            && Self::presentation_support(instance.clone(), p.clone(), i as u32)
                                .unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })?;

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_features: device_features,
                ..Default::default()
            },
        )
        .ok()?;

        let queue = queues.next()?;

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let sizes = device
            .physical_device()
            .memory_properties()
            .memory_types
            .iter()
            .map(|m| {
                device.physical_device().memory_properties().memory_heaps[m.heap_index as usize]
                    .size
            })
            .collect::<Vec<_>>();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(
            device.clone(),
            GenericMemoryAllocatorCreateInfo {
                block_sizes: &sizes,
                ..Default::default()
            },
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        Some(Self {
            instance,
            queue_family_index,
            device,
            queue,
            command_buffer_allocator,
            memory_allocator,
            descriptor_set_allocator,
        })
    }

    pub fn instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }

    pub fn command_buffer_allocator(&self) -> Arc<StandardCommandBufferAllocator> {
        self.command_buffer_allocator.clone()
    }

    pub fn memory_allocator(&self) -> Arc<GenericMemoryAllocator<FreeListAllocator>> {
        self.memory_allocator.clone()
    }

    pub fn descriptor_set_allocator(&self) -> Arc<StandardDescriptorSetAllocator> {
        self.descriptor_set_allocator.clone()
    }

    #[allow(unused_variables)]
    fn presentation_support(
        instance: Arc<Instance>,
        physical_device: Arc<PhysicalDevice>,
        queue_family_index: u32,
    ) -> Result<bool, Validated<HandleError>> {
        #[cfg(target_os = "linux")]
        {
            // TODO: check for Wayland or X11
            Ok(true)
        }

        #[cfg(target_os = "windows")]
        {
            let fns = instance.fns();
            let support = unsafe {
                (fns.khr_win32_surface
                    .get_physical_device_win32_presentation_support_khr)(
                    physical_device.handle(),
                    queue_family_index,
                )
            };

            Ok(support != 0)
        }

        #[cfg(target_os = "macos")]
        {
            Ok(true)
        }

        #[cfg(target_os = "android")]
        {
            Ok(true)
        }
    }
}
