use crate::vulkan_instance::VulkanInstance;
use bevy::prelude::{Component, GlobalTransform, PerspectiveProjection};
use bevy::render::camera::CameraProjection;
use std::ops::Deref;
use vulkano::buffer::{
    AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
    PrimaryCommandBufferAbstract,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::{GpuFuture, HostAccessError};
use vulkano::{Validated, ValidationError};

#[derive(Debug)]
pub enum CameraError {
    Allocation(Validated<AllocateBufferError>),
    HostAccess(HostAccessError),
    Validation(Box<ValidationError>),
}

impl From<Validated<AllocateBufferError>> for CameraError {
    fn from(value: Validated<AllocateBufferError>) -> Self {
        Self::Allocation(value)
    }
}

impl From<HostAccessError> for CameraError {
    fn from(value: HostAccessError) -> Self {
        Self::HostAccess(value)
    }
}

impl From<Box<ValidationError>> for CameraError {
    fn from(value: Box<ValidationError>) -> Self {
        Self::Validation(value)
    }
}

#[derive(Clone, Debug, Component)]
pub struct VoxelCamera {
    pub projection: PerspectiveProjection,
    pub aperture: f32,
    pub focus_distance: f32,
    pub samples: u32,
    pub bounces: u32,
}

impl VoxelCamera {
    pub fn new(
        projection: PerspectiveProjection,
        aperture: f32,
        focus_distance: f32,
        samples: u32,
        bounces: u32,
    ) -> Self {
        Self {
            projection,
            aperture,
            focus_distance,
            samples,
            bounces,
        }
    }

    pub fn update_aspect_ratio(&mut self, width: f32, height: f32) {
        self.projection.aspect_ratio = width / height;
    }
}

impl Default for VoxelCamera {
    fn default() -> Self {
        Self::new(
            PerspectiveProjection {
                aspect_ratio: 16.0 / 9.0,
                fov: 90.0f32.to_radians(),
                near: 0.001,
                far: 10000.0,
            },
            0.0,
            3.4,
            5,
            10,
        )
    }
}

#[derive(Component)]
pub struct VoxelCameraData {
    camera_gpu_buffer: Subbuffer<RayCamera>,
    camera_cpu_buffer: Subbuffer<RayCamera>,
}

impl VoxelCameraData {
    pub fn new(
        camera: &VoxelCamera,
        transform: &GlobalTransform,
        vulkan_instance: &VulkanInstance,
    ) -> Result<Self, Validated<AllocateBufferError>> {
        let ray_camera = RayCamera::from((camera, transform));
        let camera_gpu_buffer = Buffer::new_sized(
            vulkan_instance.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;
        let camera_cpu_buffer = Buffer::from_data(
            vulkan_instance.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            ray_camera,
        )?;

        Ok(Self {
            camera_gpu_buffer,
            camera_cpu_buffer,
        })
    }

    pub fn update_buffer(
        &mut self,
        vulkan_instance: &VulkanInstance,
        camera: &VoxelCamera,
        transform: &GlobalTransform,
    ) -> Result<(), CameraError> {
        let builder = AutoCommandBufferBuilder::primary(
            vulkan_instance.command_buffer_allocator(),
            vulkan_instance.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let ray_camera = RayCamera::from((camera, transform));
        self.camera_cpu_buffer = Buffer::from_data(
            vulkan_instance.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            ray_camera,
        )?;
        // builder.copy_buffer(CopyBufferInfo::buffers(
        //     self.camera_cpu_buffer.clone(),
        //     self.camera_gpu_buffer.clone(),
        // ))?;

        let cmd = builder.build().unwrap();
        cmd.execute(vulkan_instance.queue())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        Ok(())
    }

    // TODO: absolutely fix this
    pub fn camera_gpu_buffer(&self) -> Subbuffer<RayCamera> {
        self.camera_cpu_buffer.clone()
    }
}

#[derive(Debug, BufferContents, Copy, Clone)]
#[repr(C)]
pub struct RayCamera {
    view_proj: [[f32; 4]; 4],
    view_inverse: [[f32; 4]; 4],
    proj_inverse: [[f32; 4]; 4],
    aperture: f32,
    focus_distance: f32,
    samples: u32,
    bounces: u32,
}

impl<C: Deref<Target = VoxelCamera>, T: Deref<Target = GlobalTransform>> From<(C, T)>
    for RayCamera
{
    fn from(camera_and_transform: (C, T)) -> Self {
        let (camera, transform) = camera_and_transform;
        let projection = camera.projection.get_clip_from_view();
        let view = transform.compute_matrix();
        RayCamera {
            view_proj: (projection * view).to_cols_array_2d(),
            view_inverse: view.inverse().to_cols_array_2d(),
            proj_inverse: projection.inverse().to_cols_array_2d(),
            aperture: camera.aperture,
            focus_distance: camera.focus_distance,
            samples: camera.samples,
            bounces: camera.bounces,
        }
    }
}
