use crate::vulkan::device::VulkanDevice;
use ash::prelude::VkResult;
use ash::vk::ShaderModule;
use ash::{Device, vk};
use std::path::Path;

pub struct VulkanShader {
    device: Device,
    pub(crate) shader_module: ShaderModule,
}

impl VulkanShader {
    pub fn new(path: impl AsRef<Path>, device: &VulkanDevice) -> VkResult<Self> {
        if !path.as_ref().exists() {
            panic!("no shader found in {:?}", path.as_ref());
        }
        if path.as_ref().is_dir() {
            panic!(
                "only files are supported, {:?} is a directory",
                path.as_ref()
            );
        }

        let content = std::fs::read(path).unwrap();

        Self::new_with_content(content, device)
    }

    pub fn new_with_content(content: impl Into<Vec<u8>>, device: &VulkanDevice) -> VkResult<Self> {
        let content = content.into();
        let len = content.len();

        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: len,
            p_code: content.as_ptr().cast(),
            ..Default::default()
        };

        let shader_module = unsafe {
            device
                .device()
                .create_shader_module(&shader_module_create_info, None)?
        };

        Ok(Self {
            shader_module,
            device: device.device().clone(),
        })
    }
}

impl Drop for VulkanShader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.shader_module, None);
        }
    }
}
