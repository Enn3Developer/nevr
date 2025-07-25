use crate::context::{
    Camera, GraphicsContext, RayCamera, build_acceleration_structure_voxels,
    build_top_level_acceleration_structure,
};
use crate::voxel::{Voxel, VoxelLibrary};
use glam::{Mat4, Vec3};
use std::sync::Arc;
use vulkano::acceleration_structure::{
    AabbPositions, AccelerationStructure, AccelerationStructureInstance,
};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use winit::event::ElementState;
use winit::keyboard::KeyCode;

pub trait Scene {
    fn updated_voxels(&mut self) -> bool;
    fn get_blocks(&self) -> &[(u32, Vec3)];
    fn update(&mut self, ctx: &mut RunContext, delta: f32);
    fn input(&mut self, ctx: &mut RunContext, delta: f32, key_code: KeyCode, state: ElementState);
}

pub enum RunCommand {
    MoveCamera(Vec3),
}

pub struct RunContext {
    commands: Vec<RunCommand>,
}

impl RunContext {
    fn new() -> Self {
        Self { commands: vec![] }
    }

    pub fn add_command(&mut self, command: RunCommand) {
        self.commands.push(command);
    }

    pub fn move_camera(&mut self, movement: Vec3) {
        self.add_command(RunCommand::MoveCamera(movement));
    }
}

pub struct SceneManager {
    current_scene: Box<dyn Scene>,
    voxel_library: VoxelLibrary,
    camera: Camera,
    update_camera: bool,
    camera_buffer: Option<Subbuffer<RayCamera>>,
    blas: Option<Arc<AccelerationStructure>>,
    tlas: Option<Arc<AccelerationStructure>>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    intersect_descriptor_set: Option<Arc<DescriptorSet>>,
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

        Self {
            voxel_library,
            camera,
            update_camera: true,
            current_scene: scene,
            camera_buffer: None,
            blas: None,
            tlas: None,
            descriptor_set: None,
            intersect_descriptor_set: None,
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

        ctx.draw(
            self.descriptor_set.clone().unwrap(),
            self.intersect_descriptor_set.clone().unwrap(),
        );
    }

    pub fn input(&mut self, key_code: KeyCode, state: ElementState, delta: f32) {
        let mut ctx = RunContext::new();
        self.current_scene.input(&mut ctx, delta, key_code, state);

        for command in ctx.commands {
            match command {
                RunCommand::MoveCamera(movement) => {
                    self.update_camera = true;
                    let (scale, rotation, mut translation) =
                        self.camera.view.to_scale_rotation_translation();
                    translation -= rotation * (rotation * movement);
                    self.camera.view =
                        Mat4::from_scale_rotation_translation(scale, rotation, translation);
                    self.camera.frame = 0;
                }
            }
        }
    }

    pub fn update(&mut self) {
        self.camera.frame += 1;
        self.update_camera = true;
    }
}
