use crate::engine::context::GraphicsContext;
use vulkano::Version;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{WindowAttributes, WindowId};

pub struct App {
    context: Option<GraphicsContext>,
    window_attributes: WindowAttributes,
    app_name: String,
    app_version: Version,
}

impl App {
    pub fn new(
        app_name: impl Into<String>,
        app_version: impl Into<Version>,
        window_attributes: WindowAttributes,
    ) -> Self {
        Self {
            window_attributes,
            app_name: app_name.into(),
            app_version: app_version.into(),
            context: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.context = GraphicsContext::new(
            &self.app_name,
            &self.app_version,
            event_loop,
            self.window_attributes.clone(),
        );
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => {}
        }
    }
}
