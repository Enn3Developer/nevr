use crate::camera::{Camera, VoxelCamera};
use crate::context::{GraphicsContext, Light};
use crate::voxel::{VoxelLibrary, VoxelMaterial, VoxelType};
use crate::vulkan_instance::VulkanInstance;
use crate::world::VoxelWorld;
use egui_winit_vulkano::Gui;
use std::cell::RefCell;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;
use winit::window::CursorGrabMode;

pub trait Scene {
    fn updated_voxels(&mut self) -> bool;
    fn get_blocks(&self) -> &[(u32, glm::Vec3)];
    fn update(&mut self, ctx: &RunContext, delta: f32);
    fn ui(&mut self, gui: &mut Gui, ctx: &RunContext, delta: f32);
}

enum RunCommand {
    SetCamera(Camera),
    MoveCamera(glm::Vec3, f32),
    RotateCamera(f32, f32),
    CameraConfig(f32, f32),
    Exit,
    SkyColor(glm::Vec3),
    AmbientLight(glm::Vec4),
    LightDirection(glm::Vec4),
    ChangeScene(Box<dyn Scene>),
    GrabCursor(bool),
    Samples(u32),
    Bounces(u32),
    VoxelMaterial(u32, VoxelMaterial),
    VoxelType(u32, VoxelType),
}

pub struct RunContext<'a> {
    commands: RefCell<Vec<RunCommand>>,
    input_state: &'a InputState,
}

impl<'a> RunContext<'a> {
    fn new(input_state: &'a InputState) -> Self {
        Self {
            input_state,
            commands: RefCell::new(vec![]),
        }
    }

    pub fn input_state(&self) -> &InputState {
        self.input_state
    }

    fn add_command(&self, command: RunCommand) {
        self.commands.borrow_mut().push(command);
    }

    pub fn set_camera(&self, camera: Camera) {
        self.add_command(RunCommand::SetCamera(camera));
    }

    pub fn move_camera(&self, movement: glm::Vec3, speed: f32) {
        self.add_command(RunCommand::MoveCamera(movement, speed));
    }

    pub fn rotate_camera(&self, yaw: f32, pitch: f32) {
        self.add_command(RunCommand::RotateCamera(yaw, pitch));
    }

    pub fn change_camera_config(&self, aperture: f32, focus_distance: f32) {
        self.add_command(RunCommand::CameraConfig(aperture, focus_distance));
    }

    pub fn request_exit(&self) {
        self.add_command(RunCommand::Exit);
    }

    pub fn change_sky_color(&self, color: glm::Vec3) {
        self.add_command(RunCommand::SkyColor(color));
    }

    pub fn change_ambient_light(&self, color: glm::Vec4) {
        self.add_command(RunCommand::AmbientLight(color));
    }

    pub fn change_light_direction(&self, direction: glm::Vec4) {
        self.add_command(RunCommand::LightDirection(direction.normalize()));
    }

    pub fn change_scene(&self, scene: impl Scene + 'static) {
        self.add_command(RunCommand::ChangeScene(Box::new(scene)));
    }

    pub fn grab_cursor(&self, grab_cursor: bool) {
        self.add_command(RunCommand::GrabCursor(grab_cursor));
    }

    pub fn set_samples(&self, samples: u32) {
        self.add_command(RunCommand::Samples(samples));
    }

    pub fn set_bounces(&self, bounces: u32) {
        self.add_command(RunCommand::Bounces(bounces));
    }

    pub fn add_voxel_material(&self, id: impl Into<u32>, material: VoxelMaterial) {
        self.add_command(RunCommand::VoxelMaterial(id.into(), material));
    }

    pub fn add_voxel_type(&self, id: impl Into<u32>, voxel_type: VoxelType) {
        self.add_command(RunCommand::VoxelType(id.into(), voxel_type));
    }
}

pub struct InputState {
    keys: Vec<KeyCode>,
    buttons: Vec<MouseButton>,
    mouse_movement: glm::Vec2,
}

impl InputState {
    fn new() -> Self {
        Self {
            keys: vec![],
            buttons: vec![],
            mouse_movement: glm::Vec2::zeros(),
        }
    }

    pub fn is_key_pressed(&self, key_code: KeyCode) -> bool {
        self.keys.contains(&key_code)
    }

    pub fn is_button_pressed(&self, button: MouseButton) -> bool {
        self.buttons.contains(&button)
    }

    pub fn mouse_movement(&self) -> glm::Vec2 {
        self.mouse_movement
    }
}

pub struct SceneManager {
    vulkan_instance: Arc<VulkanInstance>,
    current_scene: Box<dyn Scene>,
    voxel_world: VoxelWorld,
    camera: VoxelCamera,
    sky_color: glm::Vec3,
    light: Light,
    descriptor_set: Option<Arc<DescriptorSet>>,
    intersect_descriptor_set: Option<Arc<DescriptorSet>>,
    input_state: InputState,
}

impl SceneManager {
    pub fn new(
        vulkan_instance: Arc<VulkanInstance>,
        scene: Box<dyn Scene>,
        voxel_library: VoxelLibrary,
    ) -> Self {
        let mut proj = glm::Mat4::new_perspective(16.0 / 9.0, 90.0_f32.to_radians(), 0.001, 1000.0);
        proj.m22 *= -1.0;

        let camera = Camera::new(
            proj,
            glm::Vec3::new(-8.0, 2.0, 2.0),
            glm::Vec2::new(0.0, 0.0),
            0.0,
            3.4,
            20,
            10,
        );

        let camera = VoxelCamera::new(camera, vulkan_instance.clone()).unwrap();

        let light_direction = glm::Vec4::new(-0.75, -1.0, 0.0, 0.0).normalize();
        let light = Light {
            ambient_light: [0.3, 0.3, 0.3, 0.0],
            light_direction: [
                light_direction.x,
                light_direction.y,
                light_direction.z,
                light_direction.w,
            ],
        };

        let voxel_world = VoxelWorld::new(vulkan_instance.clone(), voxel_library);

        Self {
            vulkan_instance,
            voxel_world,
            camera,
            light,
            current_scene: scene,
            descriptor_set: None,
            intersect_descriptor_set: None,
            input_state: InputState::new(),
            sky_color: glm::Vec3::new(0.5, 0.7, 1.0),
        }
    }

    pub(crate) fn draw(&mut self, ctx: &mut GraphicsContext) {
        if self.camera.update_camera() {
            self.camera
                .update_buffer(ctx.builder.as_mut().unwrap())
                .unwrap();

            if self.voxel_world.has_tlas() {
                self.descriptor_set = Some(
                    DescriptorSet::new(
                        self.vulkan_instance.descriptor_set_allocator(),
                        ctx.pipeline_layout.set_layouts()[0].clone(),
                        [
                            WriteDescriptorSet::acceleration_structure(0, self.voxel_world.tlas()),
                            WriteDescriptorSet::buffer(1, self.camera.camera_gpu_buffer()),
                        ],
                        [],
                    )
                    .unwrap(),
                );
            }
        }

        if self.current_scene.updated_voxels() {
            let (material_data, voxel_data) = self
                .voxel_world
                .update(self.current_scene.get_blocks().to_vec());

            self.descriptor_set = Some(
                DescriptorSet::new(
                    self.vulkan_instance.descriptor_set_allocator(),
                    ctx.pipeline_layout.set_layouts()[0].clone(),
                    [
                        WriteDescriptorSet::acceleration_structure(0, self.voxel_world.tlas()),
                        WriteDescriptorSet::buffer(1, self.camera.camera_gpu_buffer()),
                    ],
                    [],
                )
                .unwrap(),
            );

            self.intersect_descriptor_set = Some(
                DescriptorSet::new(
                    self.vulkan_instance.descriptor_set_allocator(),
                    ctx.pipeline_layout.set_layouts()[2].clone(),
                    [
                        WriteDescriptorSet::buffer(0, voxel_data.clone()),
                        WriteDescriptorSet::buffer(1, material_data.clone()),
                    ],
                    [],
                )
                .unwrap(),
            );
        }

        let sky_color_buffer = Buffer::from_data(
            self.vulkan_instance.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [self.sky_color.x, self.sky_color.y, self.sky_color.z],
        )
        .unwrap();

        let light_buffer = Buffer::from_data(
            self.vulkan_instance.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.light,
        )
        .unwrap();

        let sky_color_descriptor_set = DescriptorSet::new(
            self.vulkan_instance.descriptor_set_allocator(),
            ctx.pipeline_layout.set_layouts()[3].clone(),
            [
                WriteDescriptorSet::buffer(0, sky_color_buffer),
                WriteDescriptorSet::buffer(1, light_buffer),
            ],
            [],
        )
        .unwrap();

        ctx.draw(
            self.descriptor_set.clone().unwrap(),
            self.intersect_descriptor_set.clone().unwrap(),
            sky_color_descriptor_set,
        );
    }

    pub fn ui(&mut self, graphics_ctx: &mut GraphicsContext, delta: f32) {
        let ctx = RunContext::new(&self.input_state);
        self.current_scene.ui(&mut graphics_ctx.gui, &ctx, delta);

        self.parse_commands(ctx.commands.take(), graphics_ctx, delta);
    }

    pub fn input(&mut self, key_code: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                if !self.input_state.is_key_pressed(key_code) {
                    self.input_state.keys.push(key_code)
                }
            }
            ElementState::Released => {
                if let Some((i, _)) = self
                    .input_state
                    .keys
                    .iter()
                    .enumerate()
                    .find(|(_i, k)| **k == key_code)
                {
                    self.input_state.keys.remove(i);
                }
            }
        }
    }

    pub fn input_mouse(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                if !self.input_state.is_button_pressed(button) {
                    self.input_state.buttons.push(button)
                }
            }
            ElementState::Released => {
                if let Some((i, _)) = self
                    .input_state
                    .buttons
                    .iter()
                    .enumerate()
                    .find(|(_i, b)| **b == button)
                {
                    self.input_state.buttons.remove(i);
                }
            }
        }
    }

    pub fn input_mouse_movement(&mut self, delta: (f32, f32)) {
        self.input_state.mouse_movement += glm::Vec2::new(delta.0, delta.1);
    }

    pub fn update(&mut self, graphics_ctx: &mut GraphicsContext, delta: f32) -> bool {
        let ctx = RunContext::new(&self.input_state);
        self.current_scene.update(&ctx, delta);

        let result = self.parse_commands(ctx.commands.take(), graphics_ctx, delta);

        self.input_state.mouse_movement = glm::Vec2::zeros();
        result
    }

    fn parse_commands(
        &mut self,
        commands: Vec<RunCommand>,
        graphics_ctx: &mut GraphicsContext,
        delta: f32,
    ) -> bool {
        let mut new_scene = None;

        for command in commands {
            match command {
                RunCommand::SetCamera(camera) => self.camera.set_camera(camera),
                RunCommand::MoveCamera(movement, speed) => {
                    let movement = movement.normalize();
                    let mut position = self.camera.position();
                    position += speed * delta * movement.z * self.camera.front();
                    position += speed
                        * delta
                        * movement.x
                        * glm::normalize(&glm::cross(&self.camera.front(), &self.camera.up()));
                    self.camera.set_position(position);
                }
                #[allow(unused_variables)]
                RunCommand::RotateCamera(yaw, pitch) => {
                    let direction = glm::Vec3::new(
                        yaw.cos() * pitch.cos(),
                        pitch.sin(),
                        yaw.sin() * pitch.cos(),
                    )
                    .normalize();

                    self.camera.set_front(direction);
                }
                RunCommand::CameraConfig(aperture, focus_distance) => {
                    self.camera.set_aperture(aperture);
                    self.camera.set_focus_distance(focus_distance);
                }
                RunCommand::Exit => return true,
                RunCommand::SkyColor(color) => self.sky_color = color,
                RunCommand::AmbientLight(color) => {
                    self.light.ambient_light = [color.x, color.y, color.z, color.w]
                }
                RunCommand::LightDirection(direction) => {
                    self.light.light_direction =
                        [direction.x, direction.y, direction.z, direction.w]
                }
                RunCommand::ChangeScene(scene) => new_scene = Some(scene),
                RunCommand::GrabCursor(grab_cursor) => {
                    graphics_ctx.window.set_cursor_visible(!grab_cursor);
                    graphics_ctx
                        .window
                        .set_cursor_grab(if grab_cursor {
                            CursorGrabMode::Locked
                        } else {
                            CursorGrabMode::None
                        })
                        .unwrap();
                }
                RunCommand::Samples(samples) => {
                    self.camera.set_samples(samples);
                }
                RunCommand::Bounces(bounces) => {
                    self.camera.set_bounces(bounces);
                }
                RunCommand::VoxelMaterial(id, material) => {
                    self.voxel_world.new_material(id, material)
                }
                RunCommand::VoxelType(id, voxel_type) => self.voxel_world.new_type(id, voxel_type),
            }
        }

        if let Some(scene) = new_scene {
            self.current_scene = scene;
            self.camera.set_update_camera(true);
        }

        false
    }
}
