use crate::context::{
    Camera, GraphicsContext, Light, RayCamera, build_acceleration_structure_voxels,
    build_top_level_acceleration_structure,
};
use crate::voxel::{Voxel, VoxelLibrary};
use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
use std::cell::RefCell;
use std::sync::Arc;
use vulkano::acceleration_structure::{
    AabbPositions, AccelerationStructure, AccelerationStructureInstance,
};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

pub trait Scene {
    fn updated_voxels(&mut self) -> bool;
    fn get_blocks(&self) -> &[(u32, Vec3)];
    fn update(&mut self, ctx: &RunContext, delta: f32);
}

enum RunCommand {
    SetCamera(Camera),
    MoveCamera(Vec3, f32),
    RotateCamera(f32, f32),
    CameraConfig(f32, f32),
    Exit,
    SkyColor(Vec3),
    AmbientLight(Vec4),
    LightDirection(Vec4),
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

    pub fn move_camera(&self, movement: Vec3, speed: f32) {
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

    pub fn change_sky_color(&self, color: Vec3) {
        self.add_command(RunCommand::SkyColor(color));
    }

    pub fn change_ambient_light(&self, color: Vec4) {
        self.add_command(RunCommand::AmbientLight(color));
    }

    pub fn change_light_direction(&self, mut direction: Vec4) {
        if !direction.is_normalized() {
            direction = direction.normalize_or_zero();
        }

        self.add_command(RunCommand::LightDirection(direction));
    }
}

pub struct InputState {
    keys: Vec<KeyCode>,
    buttons: Vec<MouseButton>,
    mouse_movement: Vec2,
}

impl InputState {
    fn new() -> Self {
        Self {
            keys: vec![],
            buttons: vec![],
            mouse_movement: Vec2::ZERO,
        }
    }

    pub fn is_key_pressed(&self, key_code: KeyCode) -> bool {
        self.keys.contains(&key_code)
    }

    pub fn is_button_pressed(&self, button: MouseButton) -> bool {
        self.buttons.contains(&button)
    }

    pub fn mouse_movement(&self) -> Vec2 {
        self.mouse_movement
    }
}

pub struct SceneManager {
    current_scene: Box<dyn Scene>,
    voxel_library: VoxelLibrary,
    camera: Camera,
    update_camera: bool,
    sky_color: Vec3,
    light: Light,
    camera_buffer: Option<Subbuffer<RayCamera>>,
    blas: Option<Arc<AccelerationStructure>>,
    tlas: Option<Arc<AccelerationStructure>>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    intersect_descriptor_set: Option<Arc<DescriptorSet>>,
    input_state: InputState,
}

impl SceneManager {
    pub fn new(scene: Box<dyn Scene>, voxel_library: VoxelLibrary) -> Self {
        let proj = Mat4::perspective_rh(90.0_f32.to_radians(), 16.0 / 9.0, 0.001, 1000.0);
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        );

        let camera = Camera {
            aperture: 0.0,
            focus_distance: 3.4,
            projection: proj,
            view,
            frame: 0,
        };
        let light = Light {
            ambient_light: Vec4::new(0.3, 0.3, 0.3, 0.0).to_array(),
            light_direction: Vec4::new(-0.75, -1.0, 0.0, 0.0).normalize().to_array(),
        };

        Self {
            voxel_library,
            camera,
            light,
            update_camera: true,
            current_scene: scene,
            camera_buffer: None,
            blas: None,
            tlas: None,
            descriptor_set: None,
            intersect_descriptor_set: None,
            input_state: InputState::new(),
            sky_color: Vec3::new(0.5, 0.7, 1.0),
        }
    }

    pub(crate) fn draw(&mut self, ctx: &mut GraphicsContext) {
        if self.update_camera {
            self.update_camera = false;

            let ray_camera = Arc::new(RayCamera::from(&self.camera));
            println!("frames: {}", ray_camera.frame);

            self.camera_buffer = Some(
                Buffer::from_data(
                    ctx.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    *ray_camera,
                )
                .unwrap(),
            );

            if let Some(tlas) = self.tlas.as_ref() {
                self.descriptor_set = Some(
                    DescriptorSet::new(
                        ctx.descriptor_set_allocator.clone(),
                        ctx.pipeline_layout.set_layouts()[0].clone(),
                        [
                            WriteDescriptorSet::acceleration_structure(0, tlas.clone()),
                            WriteDescriptorSet::buffer(
                                1,
                                self.camera_buffer.as_ref().unwrap().clone(),
                            ),
                        ],
                        [],
                    )
                    .unwrap(),
                );
            }
        }

        if self.current_scene.updated_voxels() {
            let blocks = self
                .current_scene
                .get_blocks()
                .iter()
                .map(|(id, pos)| self.voxel_library.create_block(*id, *pos).unwrap())
                .collect::<Vec<_>>();

            let voxels = blocks
                .iter()
                .map(|block| block.voxel_array())
                .flatten()
                .collect::<Vec<Voxel>>();

            let voxel_data = Buffer::from_iter(
                ctx.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                voxels.clone(),
            )
            .unwrap();

            let material_data = Buffer::from_iter(
                ctx.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                self.voxel_library.materials.clone(),
            )
            .unwrap();

            let voxel_buffer = Buffer::from_iter(
                ctx.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                voxels.clone().into_iter().map(|v| AabbPositions::from(v)),
            )
            .unwrap();

            // Build the bottom-level acceleration structure and then the top-level acceleration
            // structure. Acceleration structures are used to accelerate ray tracing. The bottom-level
            // acceleration structure contains the geometry data. The top-level acceleration structure
            // contains the instances of the bottom-level acceleration structures. In our shader, we
            // will trace rays against the top-level acceleration structure.
            self.blas = Some(unsafe {
                build_acceleration_structure_voxels(
                    &voxel_buffer,
                    ctx.memory_allocator.clone(),
                    ctx.command_buffer_allocator.clone(),
                    ctx.device.clone(),
                    ctx.queue.clone(),
                )
            });
            self.tlas = Some(unsafe {
                build_top_level_acceleration_structure(
                    vec![AccelerationStructureInstance {
                        acceleration_structure_reference: self
                            .blas
                            .as_ref()
                            .unwrap()
                            .device_address()
                            .into(),
                        ..Default::default()
                    }],
                    ctx.memory_allocator.clone(),
                    ctx.command_buffer_allocator.clone(),
                    ctx.device.clone(),
                    ctx.queue.clone(),
                )
            });
            self.descriptor_set = Some(
                DescriptorSet::new(
                    ctx.descriptor_set_allocator.clone(),
                    ctx.pipeline_layout.set_layouts()[0].clone(),
                    [
                        WriteDescriptorSet::acceleration_structure(
                            0,
                            self.tlas.as_ref().unwrap().clone(),
                        ),
                        WriteDescriptorSet::buffer(1, self.camera_buffer.as_ref().unwrap().clone()),
                    ],
                    [],
                )
                .unwrap(),
            );

            self.intersect_descriptor_set = Some(
                DescriptorSet::new(
                    ctx.descriptor_set_allocator.clone(),
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
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.sky_color.to_array(),
        )
        .unwrap();

        let light_buffer = Buffer::from_data(
            ctx.memory_allocator.clone(),
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
            ctx.descriptor_set_allocator.clone(),
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
        self.input_state.mouse_movement += Vec2::new(delta.0, delta.1);
    }

    pub fn update(&mut self, delta: f32) -> bool {
        self.camera.frame += 1;
        self.update_camera = true;

        let ctx = RunContext::new(&self.input_state);
        self.current_scene.update(&ctx, delta);
        for command in ctx.commands.take() {
            match command {
                RunCommand::SetCamera(camera) => {
                    self.camera = camera;
                    self.update_camera = true;
                }
                RunCommand::MoveCamera(movement, speed) => {
                    let movement = movement.normalize_or_zero();
                    self.update_camera = true;
                    let (scale, rotation, mut translation) =
                        self.camera.view.to_scale_rotation_translation();
                    translation -= (rotation * (rotation * movement)) * speed * delta;
                    self.camera.view =
                        Mat4::from_scale_rotation_translation(scale, rotation, translation);
                    self.camera.frame = 0;
                }
                RunCommand::RotateCamera(yaw, pitch) => {
                    self.update_camera = true;
                    self.camera.frame = 0;
                    let (scale, rotation, translation) =
                        self.camera.view.to_scale_rotation_translation();

                    let rot_mat = Mat3::from_quat(rotation);
                    let yaw_mat = Mat3::from_axis_angle(Vec3::Y, yaw);
                    let pitch_mat = Mat3::from_axis_angle(Vec3::X, pitch);

                    let final_mat = pitch_mat * yaw_mat * rot_mat;

                    self.camera.view = Mat4::from_scale_rotation_translation(
                        scale,
                        Quat::from_mat3(&final_mat),
                        translation,
                    );
                }
                RunCommand::CameraConfig(aperture, focus_distance) => {
                    self.update_camera = true;
                    self.camera.frame = 0;
                    self.camera.aperture = aperture;
                    self.camera.focus_distance = focus_distance;
                }
                RunCommand::Exit => return true,
                RunCommand::SkyColor(color) => self.sky_color = color,
                RunCommand::AmbientLight(color) => self.light.ambient_light = color.to_array(),
                RunCommand::LightDirection(direction) => {
                    self.light.light_direction = direction.to_array()
                }
            }
        }

        self.input_state.mouse_movement = Vec2::ZERO;
        false
    }
}
