use crate::engine::context::GraphicsContext;
use crate::engine::vulkan::{
    Vulkan, VulkanApplicationInfo, VulkanInstanceCreateInfo, VulkanVersion,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{WindowAttributes, WindowId};

pub struct App {
    vulkan: Vulkan,
    context: Option<GraphicsContext>,
    window_attributes: WindowAttributes,
}

impl App {
    pub fn new(
        app_name: impl AsRef<str>,
        app_version: impl Into<VulkanVersion>,
        window_attributes: WindowAttributes,
    ) -> Self {
        let vulkan = Vulkan::new(
            VulkanInstanceCreateInfo::default()
                .with_app_info(
                    VulkanApplicationInfo::default()
                        .with_app_version(app_version)
                        .with_application_name(app_name)
                        .with_engine_version((0, 1, 0))
                        .with_engine_name("NEVR"),
                )
                .with_extensions(vec!["VK_KHR_surface"])
                .enable_debug(),
        )
        .unwrap();

        Self {
            vulkan,
            window_attributes,
            context: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.context =
            GraphicsContext::new(&self.vulkan, event_loop, self.window_attributes.clone());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        todo!()
    }
}
