use crate::vulkan::Vulkan;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::surface::VulkanSurface;
use ash::khr::swapchain;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::{
    ComponentMapping, ComponentSwizzle, CompositeAlphaFlagsKHR, Flags, ImageAspectFlags,
    ImageLayout, ImageSubresourceRange, ImageUsageFlags, ImageViewType, SharingMode,
};

pub struct VulkanSwapchain {
    loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
}

impl VulkanSwapchain {
    pub fn new(
        vulkan: &Vulkan,
        physical_device: &vk::PhysicalDevice,
        device: &mut VulkanDevice,
        surface: &VulkanSurface,
    ) -> VkResult<Self> {
        let surface_capabilities = surface.get_surface_capabilities(physical_device)?;
        let surface_formats = surface.get_surface_formats(physical_device)?;
        let present_modes = surface.get_present_modes(physical_device)?;

        let loader = swapchain::Device::new(vulkan.instance(), device.device());
        let create_info = vk::SwapchainCreateInfoKHR {
            surface: *surface.surface(),
            min_image_count: surface_capabilities.min_image_count + 1,
            image_format: surface_formats[0].format,
            image_color_space: surface_formats[0].color_space,
            image_extent: surface_capabilities.max_image_extent,
            image_array_layers: 1,
            image_usage: ImageUsageFlags::from_raw(Flags::from(
                ImageLayout::TRANSFER_DST_OPTIMAL.as_raw() as u32,
            )),
            image_sharing_mode: SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: device.queue_family_index(),
            pre_transform: surface_capabilities.current_transform,
            composite_alpha: CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: present_modes[0],
            clipped: true.into(),
            ..Default::default()
        };

        let swapchain = unsafe { loader.create_swapchain(&create_info, None)? };

        let images = unsafe { loader.get_swapchain_images(swapchain)? };

        let mut image_views = vec![];

        for i in 0..images.len() {
            let image_view_create_info = vk::ImageViewCreateInfo {
                image: images[i],
                view_type: ImageViewType::TYPE_2D,
                format: surface_formats[0].format,
                components: ComponentMapping::default()
                    .a(ComponentSwizzle::IDENTITY)
                    .r(ComponentSwizzle::IDENTITY)
                    .g(ComponentSwizzle::IDENTITY)
                    .b(ComponentSwizzle::IDENTITY),
                subresource_range: ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_array_layer: 0,
                    base_mip_level: 0,
                    layer_count: 1,
                    level_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };

            unsafe {
                image_views.push(
                    device
                        .device()
                        .create_image_view(&image_view_create_info, None)?,
                );
            }
        }

        device.images = images;
        device.image_views = image_views;

        Ok(Self { loader, swapchain })
    }
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
