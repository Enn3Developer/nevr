use crate::vulkan::Vulkan;
use ash::khr::surface;
use ash::prelude::VkResult;
use ash::vk::{PresentModeKHR, SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR};
use ash::{Entry, Instance, vk};
use std::ops::{Deref, DerefMut};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

pub struct VulkanSurface {
    surface: SurfaceKHR,
    loader: surface::Instance,
}

impl VulkanSurface {
    pub fn new(entry: &Entry, instance: &Instance, window: &Window) -> VkResult<Self> {
        let loader = surface::Instance::new(entry, instance);

        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle().unwrap().into(),
                window.window_handle().unwrap().into(),
                None,
            )?
        };

        Ok(Self { surface, loader })
    }

    pub fn surface(&self) -> &SurfaceKHR {
        &self.surface
    }

    pub fn loader(&self) -> &surface::Instance {
        &self.loader
    }

    pub fn get_surface_capabilities(
        &self,
        physical_device: &vk::PhysicalDevice,
    ) -> VkResult<SurfaceCapabilitiesKHR> {
        unsafe {
            self.loader
                .get_physical_device_surface_capabilities(*physical_device, self.surface)
        }
    }

    pub fn get_surface_formats(
        &self,
        physical_device: &vk::PhysicalDevice,
    ) -> VkResult<Vec<SurfaceFormatKHR>> {
        unsafe {
            self.loader
                .get_physical_device_surface_formats(*physical_device, self.surface)
        }
    }

    pub fn get_present_modes(
        &self,
        physical_device: &vk::PhysicalDevice,
    ) -> VkResult<Vec<PresentModeKHR>> {
        unsafe {
            self.loader
                .get_physical_device_surface_present_modes(*physical_device, self.surface)
        }
    }
}

impl Deref for VulkanSurface {
    type Target = SurfaceKHR;

    fn deref(&self) -> &Self::Target {
        &self.surface
    }
}

impl DerefMut for VulkanSurface {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.surface
    }
}

impl Drop for VulkanSurface {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}
