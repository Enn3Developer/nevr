use crate::engine::vulkan::Vulkan;
use crate::engine::vulkan::surface::VulkanSurface;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

pub struct GraphicsContext {
    window: Window,
    surface: VulkanSurface,
}

impl GraphicsContext {
    pub fn new(
        vulkan: &Vulkan,
        event_loop: &ActiveEventLoop,
        attributes: WindowAttributes,
    ) -> Option<Self> {
        let window = event_loop.create_window(attributes).ok()?;
        let surface = vulkan.create_surface(&window).ok()?;
        let (physical_device, queue_family_index) = vulkan.find_physical_device(&surface)?;

        Some(Self { window, surface })
    }
}
