use ash::khr::surface;
use ash::prelude::VkResult;
use ash::vk::SurfaceKHR;
use ash::{Entry, Instance};
use std::ops::{Deref, DerefMut};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

pub struct VulkanSurface {
    surface: SurfaceKHR,
    loader: surface::Instance,
}

impl VulkanSurface {
    pub fn new(entry: &Entry, instance: &Instance, window: &Window) -> VkResult<Self> {
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle().unwrap().into(),
                window.window_handle().unwrap().into(),
                None,
            )?
        };

        let loader = surface::Instance::new(entry, instance);

        Ok(Self { surface, loader })
    }

    pub fn surface(&self) -> &SurfaceKHR {
        &self.surface
    }

    pub fn loader(&self) -> &surface::Instance {
        &self.loader
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
