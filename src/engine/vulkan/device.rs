use crate::vulkan::Vulkan;
use ash::prelude::VkResult;
use ash::vk::PhysicalDevice;
use ash::{Device, vk};
use std::ffi::CString;
use std::str::FromStr;

pub struct VulkanDevice {
    device: Device,
}

impl VulkanDevice {
    pub fn new<S: AsRef<str>>(
        vulkan: &Vulkan,
        physical_device: &PhysicalDevice,
        queue_family_index: u32,
        device_extensions: impl IntoIterator<Item = S>,
        raytracing_pipeline_features: &vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
        device_feature: &vk::PhysicalDeviceFeatures,
    ) -> VkResult<Self> {
        let priorities = [1.0f32];
        let extensions = device_extensions
            .into_iter()
            .map(|e| CString::from_str(e.as_ref()).unwrap())
            .collect::<Vec<_>>();
        let raw_extensions = extensions.iter().map(|e| e.as_ptr()).collect::<Vec<_>>();

        let pointer_raytracing_pipeline_features:
            *const vk::PhysicalDeviceRayTracingPipelineFeaturesKHR = raytracing_pipeline_features;

        let device_queue_create_info = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: priorities.len() as u32,
            p_queue_priorities: priorities.as_ptr(),
            ..Default::default()
        };

        let device_create_info = vk::DeviceCreateInfo {
            p_next: pointer_raytracing_pipeline_features.cast(),
            queue_create_info_count: 1,
            p_queue_create_infos: &device_queue_create_info,
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: raw_extensions.as_ptr(),
            p_enabled_features: device_feature,
            ..Default::default()
        };

        Ok(Self {
            device: unsafe {
                vulkan
                    .instance()
                    .create_device(*physical_device, &device_create_info, None)?
            },
        })
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_device(None);
        }
    }
}
