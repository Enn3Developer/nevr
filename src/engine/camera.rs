use crate::vulkan_instance::VulkanInstance;
use bevy::prelude::Component;
use std::ops::Deref;
use std::sync::Arc;
use vulkano::buffer::{
    AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::HostAccessError;
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
    pub projection: glm::Mat4,
    pub position: glm::Vec3,
    pub front: glm::Vec3,
    pub right: glm::Vec3,
    pub up: glm::Vec3,
    pub aperture: f32,
    pub focus_distance: f32,
    pub samples: u32,
    pub bounces: u32,
}

impl VoxelCamera {
    pub fn new(
        projection: glm::Mat4,
        position: glm::Vec3,
        rotation: glm::Vec2,
        aperture: f32,
        focus_distance: f32,
        samples: u32,
        bounces: u32,
    ) -> Self {
        let direction = glm::Vec3::new(
            rotation.x.cos() * rotation.y.cos(),
            rotation.y.sin(),
            rotation.x.sin() * rotation.y.cos(),
        );
        let up = glm::Vec3::y();
        let camera_right = glm::normalize(&glm::cross(&up, &direction));
        let camera_up = glm::cross(&direction, &camera_right);

        Self {
            projection,
            position,
            front: direction,
            right: camera_right,
            up: camera_up,
            aperture,
            focus_distance,
            samples,
            bounces,
        }
    }

    pub fn view(&self) -> glm::Mat4 {
        glm::look_at(
            &self.position,
            &(self.position + self.front),
            &glm::Vec3::y(),
        )
    }
}

impl Default for VoxelCamera {
    fn default() -> Self {
        let mut proj = glm::Mat4::new_perspective(16.0 / 9.0, 90.0_f32.to_radians(), 0.001, 1000.0);
        proj.m22 *= -1.0;

        Self::new(
            proj,
            glm::Vec3::new(-8.0, 2.0, 2.0),
            glm::Vec2::new(0.0, 0.0),
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
        vulkan_instance: &VulkanInstance,
    ) -> Result<Self, Validated<AllocateBufferError>> {
        let ray_camera = RayCamera::from(camera);
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
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &VoxelCamera,
    ) -> Result<(), CameraError> {
        let ray_camera = RayCamera::from(camera);
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

impl<C: Deref<Target = VoxelCamera>> From<C> for RayCamera {
    fn from(camera: C) -> Self {
        RayCamera {
            view_proj: (camera.projection * camera.view()).data.0,
            view_inverse: glm::inverse(&camera.view()).data.0,
            proj_inverse: glm::inverse(&camera.projection).data.0,
            aperture: camera.aperture,
            focus_distance: camera.focus_distance,
            samples: camera.samples,
            bounces: camera.bounces,
        }
    }
}
