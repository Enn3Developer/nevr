use crate::context::{GraphicsContext, Voxel};
use std::sync::Arc;
use winit::event::ElementState;
use winit::keyboard::KeyCode;

pub trait Scene {
    fn get_voxels(&self) -> &[Voxel];
}

pub struct SceneManager {
    current_scene: Arc<dyn Scene>,
}

impl SceneManager {
    pub fn new(scene: Arc<dyn Scene>) -> Self {
        Self {
            current_scene: scene,
        }
    }

    pub(crate) fn draw(&self, ctx: &mut GraphicsContext) {
        ctx.draw()
    }

    pub(crate) fn get_voxels(&self) -> &[Voxel] {
        self.current_scene.get_voxels()
    }

    pub fn input(&mut self, key_code: KeyCode, state: ElementState) {}
}
