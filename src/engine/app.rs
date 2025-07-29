use crate::engine::context::GraphicsContext;
use crate::scene::{Scene, SceneManager};
use crate::voxel::VoxelLibrary;
use crate::vulkan_instance::VulkanInstance;
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
use winit::window::{WindowAttributes, WindowId};

pub struct App {
    context: Option<GraphicsContext>,
    window_attributes: WindowAttributes,
    scene_manager: SceneManager,
    last_delta: Instant,
    vulkan_instance: Arc<VulkanInstance>,
}

impl App {
    pub fn new(
        app_name: impl Into<String>,
        app_version: impl Into<Version>,
        window_attributes: WindowAttributes,
        main_scene: Box<dyn Scene>,
        voxel_library: VoxelLibrary,
    ) -> Self {
        let vulkan_instance =
            Arc::new(VulkanInstance::new(Some(app_name.into()), app_version.into()).unwrap());

        Self {
            scene_manager: SceneManager::new(vulkan_instance.clone(), main_scene, voxel_library),
            context: None,
            last_delta: Instant::now(),
            vulkan_instance,
            window_attributes,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.context = GraphicsContext::new(
            self.vulkan_instance.clone(),
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
        let ctx = self.context.as_mut().unwrap();

        let exclusive = ctx.gui.update(&event);

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
            } if !exclusive => self.scene_manager.input(key_code, state),
            WindowEvent::MouseInput { state, button, .. } if !exclusive => {
                self.scene_manager.input_mouse(button, state)
            }
            WindowEvent::RedrawRequested => {
                let current = Instant::now();
                let delta = (current - self.last_delta).as_secs_f32();
                self.last_delta = current;
                if self.scene_manager.update(ctx, delta) {
                    event_loop.exit();
                }

                self.scene_manager.ui(ctx, delta);

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
                    self.vulkan_instance.command_buffer_allocator(),
                    self.vulkan_instance.queue_family_index(),
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
                    .then_execute(self.vulkan_instance.queue(), command_buffer)
                    .unwrap();

                let after_future = ctx.gui.draw_on_image(
                    future,
                    ctx.swapchain_image_sets[image_index as usize].0.clone(),
                );

                let future = after_future
                    .then_swapchain_present(
                        self.vulkan_instance.queue(),
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
                        Some(sync::now(self.vulkan_instance.device()).boxed())
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        Some(sync::now(self.vulkan_instance.device()).boxed())
                    }
                };

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
