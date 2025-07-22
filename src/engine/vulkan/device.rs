use crate::vulkan::Vulkan;
use crate::vulkan::pipeline::VulkanPipeline;
use crate::vulkan::shader::VulkanShader;
use ash::prelude::VkResult;
use ash::vk::{
    CommandBuffer, CommandBufferLevel, CommandPool, CommandPoolCreateFlags, DescriptorPool,
    DescriptorPoolCreateFlags, DescriptorSet, DescriptorSetLayout, PhysicalDevice, Queue,
};
use ash::{Device, Instance, vk};
use std::ffi::CString;
use std::str::FromStr;

pub struct VulkanDevice {
    instance: Instance,
    device: Device,
    queue_family_index: u32,
    command_pool: CommandPool,
    queue: Queue,
    command_buffers: Vec<CommandBuffer>,
    descriptor_pool: DescriptorPool,
    descriptor_set_layout: DescriptorSetLayout,
    material_descriptor_set_layout: DescriptorSetLayout,
    descriptor_set_layouts: Vec<DescriptorSetLayout>,
    descriptor_sets: Vec<DescriptorSet>,
    pub(crate) images: Vec<vk::Image>,
    pub(crate) image_views: Vec<vk::ImageView>,
}

impl VulkanDevice {
    pub fn new<S: AsRef<str>>(
        vulkan: &Vulkan,
        physical_device: &PhysicalDevice,
        queue_family_index: u32,
        device_extensions: impl IntoIterator<Item = S>,
        raytracing_pipeline_features: &vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
        device_feature: &vk::PhysicalDeviceFeatures,
        descriptor_pool_sizes: Vec<vk::DescriptorPoolSize>,
        descriptor_set_layout_bindings: Vec<vk::DescriptorSetLayoutBinding>,
        material_descriptor_set_layout_bindings: Vec<vk::DescriptorSetLayoutBinding>,
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

        let command_pool_create_info = vk::CommandPoolCreateInfo {
            flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index,
            ..Default::default()
        };

        let device = unsafe {
            vulkan
                .instance()
                .create_device(*physical_device, &device_create_info, None)?
        };

        let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None)? };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level: CommandBufferLevel::PRIMARY,
            command_buffer_count: 16,
            ..Default::default()
        };

        let command_buffers =
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info)? };

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
            flags: DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            max_sets: 2,
            pool_size_count: descriptor_pool_sizes.len() as u32,
            p_pool_sizes: descriptor_pool_sizes.as_ptr(),
            ..Default::default()
        };

        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None)? };

        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: descriptor_set_layout_bindings.len() as u32,
            p_bindings: descriptor_set_layout_bindings.as_ptr(),
            ..Default::default()
        };

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)?
        };

        let material_descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: material_descriptor_set_layout_bindings.len() as u32,
            p_bindings: material_descriptor_set_layout_bindings.as_ptr(),
            ..Default::default()
        };

        let material_descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&material_descriptor_set_layout_create_info, None)?
        };

        let descriptor_set_layouts = vec![descriptor_set_layout, material_descriptor_set_layout];

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: descriptor_set_layouts.len() as u32,
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            ..Default::default()
        };

        let descriptor_sets =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info)? };

        Ok(Self {
            queue_family_index,
            device,
            command_pool,
            queue,
            command_buffers,
            descriptor_pool,
            descriptor_set_layout,
            material_descriptor_set_layout,
            descriptor_set_layouts,
            descriptor_sets,
            images: vec![],
            image_views: vec![],
            instance: vulkan.instance.clone(),
        })
    }

    pub fn create_pipeline(&self, shaders: [&VulkanShader; 4]) -> VkResult<VulkanPipeline> {
        VulkanPipeline::new(
            &self.instance,
            &self.descriptor_set_layouts,
            &self.device,
            shaders,
        )
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue_family_index(&self) -> &u32 {
        &self.queue_family_index
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.queue_wait_idle(self.queue);
            let _ = self.device.device_wait_idle();

            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            self.device.destroy_command_pool(self.command_pool, None);

            for image_view in &self.image_views {
                self.device.destroy_image_view(*image_view, None);
            }
            for image in &self.images {
                self.device.destroy_image(*image, None);
            }

            let _ = self
                .device
                .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.material_descriptor_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.device.destroy_device(None);
        }
    }
}
