use crate::engine::context::GraphicsContext;
use crate::scene::{Scene, SceneManager};
use crate::voxel::VoxelLibrary;
use std::sync::Arc;
use std::time::Instant;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::swapchain::{SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image};
use vulkano::sync::GpuFuture;
use vulkano::{Validated, Version, VulkanError, sync};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, WindowAttributes, WindowId};

pub struct App {
    context: Option<GraphicsContext>,
    window_attributes: WindowAttributes,
    app_name: String,
    app_version: Version,
    scene_manager: SceneManager,
    last_delta: f32,
}

impl App {
    pub fn new(
        app_name: impl Into<String>,
        app_version: impl Into<Version>,
        window_attributes: WindowAttributes,
        main_scene: Box<dyn Scene>,
        voxel_library: VoxelLibrary,
    ) -> Self {
        Self {
            window_attributes,
            app_name: app_name.into(),
            app_version: app_version.into(),
            context: None,
            scene_manager: SceneManager::new(main_scene, voxel_library),
            last_delta: 0.0,
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

        self.context
            .as_ref()
            .unwrap()
            .window
            .set_cursor_grab(CursorGrabMode::Confined)
            .unwrap();
        self.context
            .as_ref()
            .unwrap()
            .window
            .set_cursor_visible(false);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let ctx = self.context.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                println!("Closing");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                ctx.recreate_swapchain = true;
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key_code),
                        state,
                        ..
                    },
                ..
            } => self.scene_manager.input(key_code, state),
            WindowEvent::MouseInput { state, button, .. } => {
                self.scene_manager.input_mouse(button, state)
            }
            WindowEvent::RedrawRequested => {
                let start = Instant::now();
                if self.scene_manager.update(self.last_delta) {
                    event_loop.exit();
                }

                let window_size = ctx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                ctx.previous_frame.as_mut().unwrap().cleanup_finished();

                if ctx.recreate_swapchain {
                    let (new_swapchain, new_images) = ctx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..ctx.swapchain.create_info()
                        })
                        .expect("can't recreate swapchain");

                    ctx.swapchain = new_swapchain;
                    ctx.resize(new_images);
                    ctx.recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    ctx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        ctx.recreate_swapchain = true;
                        println!("out of date");
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                if suboptimal {
                    ctx.recreate_swapchain = true;
                }

                let builder = AutoCommandBufferBuilder::primary(
                    ctx.command_buffer_allocator.clone(),
                    ctx.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // RECORD COMMANDS HERE
                ctx.builder = Some(builder);
                ctx.image_index = Some(image_index);
                self.scene_manager.draw(ctx);
                // END RECORD COMMANDS

                let builder = ctx.builder.take().unwrap();
                let command_buffer = builder.build().unwrap();

                let future = ctx
                    .previous_frame
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(ctx.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        ctx.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            ctx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                ctx.previous_frame = match future.map_err(Validated::unwrap) {
                    Ok(future) => Some(future.boxed()),
                    Err(VulkanError::OutOfDate) => {
                        ctx.recreate_swapchain = true;
                        Some(sync::now(ctx.device.clone()).boxed())
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        Some(sync::now(ctx.device.clone()).boxed())
                    }
                };

                let end = Instant::now();
                self.last_delta = (end - start).as_secs_f32();
                println!(
                    "render time: {:2}ms",
                    (end - start).as_micros() as f32 / 1000.0
                );

                ctx.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => self
                .scene_manager
                .input_mouse_movement((delta.0 as f32, delta.1 as f32)),
            _ => {}
        }
    }
}
