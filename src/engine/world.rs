use crate::engine::voxel::{Voxel, VoxelBlock, VoxelLibrary, VoxelMaterial};
use crate::engine::vulkan_instance::VulkanInstance;
use bevy::prelude::{GlobalTransform, Ref, Resource};
use itertools::Itertools;
use std::iter;
use std::sync::Arc;
use vulkano::Packed24_8;
use vulkano::acceleration_structure::{
    AabbPositions, AccelerationStructure, AccelerationStructureBuildGeometryInfo,
    AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
    AccelerationStructureCreateInfo, AccelerationStructureGeometries,
    AccelerationStructureGeometryAabbsData, AccelerationStructureGeometryInstancesData,
    AccelerationStructureGeometryInstancesDataType, AccelerationStructureGeometryTrianglesData,
    AccelerationStructureInstance, AccelerationStructureType, BuildAccelerationStructureFlags,
    BuildAccelerationStructureMode,
};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::sync::GpuFuture;

#[derive(Resource, Default)]
pub struct VoxelWorld {
    blas: Vec<Arc<AccelerationStructure>>,
    tlas: Vec<Arc<AccelerationStructure>>,
}

impl VoxelWorld {
    pub fn new() -> Self {
        Self {
            blas: vec![],
            tlas: vec![],
        }
    }

    pub fn update<'a>(
        &mut self,
        blocks: impl IntoIterator<Item = (Ref<'a, VoxelBlock>, Ref<'a, GlobalTransform>)>,
        voxel_library: &VoxelLibrary,
        vulkan_instance: &VulkanInstance,
    ) -> (Subbuffer<[VoxelMaterial]>, Subbuffer<[Voxel]>) {
        // vec of voxels (Vec<Voxel>)
        let voxel_chunks = blocks
            .into_iter()
            .map(|(block, transform)| block.voxel_array(&transform).into_iter().collect_vec())
            .flatten()
            .chunks(8192)
            .into_iter()
            .map(|v| v.collect_vec())
            .collect_vec();

        let material_data = Buffer::from_iter(
            vulkan_instance.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            voxel_library.materials.clone(),
        )
        .unwrap();

        let voxel_data = Buffer::from_iter(
            vulkan_instance.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            voxel_chunks.clone().into_iter().flatten().collect_vec(),
        )
        .unwrap();

        let voxel_buffers = voxel_chunks
            .into_iter()
            .map(|voxels| {
                Buffer::from_iter(
                    vulkan_instance.memory_allocator(),
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
                    voxels.iter().map(|voxel| AabbPositions::from(*voxel)),
                )
                .unwrap()
            })
            .collect_vec();

        self.blas = voxel_buffers
            .into_iter()
            .map(|voxel_buffer| unsafe {
                build_acceleration_structure_voxels(
                    &voxel_buffer,
                    vulkan_instance.memory_allocator(),
                    vulkan_instance.command_buffer_allocator(),
                    vulkan_instance.device(),
                    vulkan_instance.queue(),
                )
            })
            .collect_vec();

        self.tlas = vec![unsafe {
            build_top_level_acceleration_structure(
                self.blas
                    .iter()
                    .enumerate()
                    .map(|(index, blas)| AccelerationStructureInstance {
                        instance_custom_index_and_mask: Packed24_8::new(index as u32, 0xFF),
                        acceleration_structure_reference: blas.device_address().into(),
                        ..AccelerationStructureInstance::default()
                    })
                    .collect_vec(),
                vulkan_instance.memory_allocator(),
                vulkan_instance.command_buffer_allocator(),
                vulkan_instance.device(),
                vulkan_instance.queue(),
            )
        }];

        (material_data, voxel_data)
    }

    pub fn tlas(&self) -> Arc<AccelerationStructure> {
        self.tlas[0].clone()
    }

    pub fn has_tlas(&self) -> bool {
        self.tlas.len() > 0
    }
}

/// A helper function to build an acceleration structure and wait for its completion.
///
/// # Safety
///
/// - If you are referencing a bottom-level acceleration structure in a top-level acceleration
///   structure, you must ensure that the bottom-level acceleration structure is kept alive.
unsafe fn build_acceleration_structure_common(
    geometries: AccelerationStructureGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: BuildAccelerationStructureMode::Build,
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let as_build_sizes_info = device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &as_build_geometry_info,
            &[primitive_count],
        )
        .unwrap();

    // We create a new scratch buffer for each acceleration structure for simplicity. You may want
    // to reuse scratch buffers if you need to build many acceleration structures.
    let scratch_buffer = Buffer::new_slice::<u8>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
        as_build_sizes_info.build_scratch_size,
    )
    .unwrap();

    let acceleration = unsafe {
        AccelerationStructure::new(
            device,
            AccelerationStructureCreateInfo {
                ty,
                ..AccelerationStructureCreateInfo::new(
                    Buffer::new_slice::<u8>(
                        memory_allocator,
                        BufferCreateInfo {
                            usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                                | BufferUsage::SHADER_DEVICE_ADDRESS,
                            ..Default::default()
                        },
                        AllocationCreateInfo::default(),
                        as_build_sizes_info.acceleration_structure_size,
                    )
                    .unwrap(),
                )
            },
        )
    }
    .unwrap();

    as_build_geometry_info.dst_acceleration_structure = Some(acceleration.clone());
    as_build_geometry_info.scratch_data = Some(scratch_buffer);

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
        ..Default::default()
    };

    // For simplicity, we build a single command buffer that builds the acceleration structure,
    // then waits for its execution to complete.
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    unsafe {
        builder.build_acceleration_structure(
            as_build_geometry_info,
            iter::once(as_build_range_info).collect(),
        )
    }
    .unwrap();

    builder
        .build()
        .unwrap()
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    acceleration
}

pub(crate) unsafe fn build_acceleration_structure_voxels(
    voxel_buffer: &Subbuffer<[AabbPositions]>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let primitive_count = voxel_buffer.len() as u32;
    let as_geometry_voxels_data = AccelerationStructureGeometryAabbsData {
        data: Some(voxel_buffer.clone().into_bytes()),
        stride: size_of::<AabbPositions>() as u32,
        ..AccelerationStructureGeometryAabbsData::default()
    };

    let geometries = AccelerationStructureGeometries::Aabbs(vec![as_geometry_voxels_data]);

    unsafe {
        build_acceleration_structure_common(
            geometries,
            primitive_count,
            AccelerationStructureType::BottomLevel,
            memory_allocator,
            command_buffer_allocator,
            device,
            queue,
        )
    }
}

pub(crate) unsafe fn build_acceleration_structure_triangles(
    primitive_count: u32,
    vertex_buffer: Subbuffer<[[i32; 3]]>,
    index_buffer: Subbuffer<[u32]>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let as_geometry_triangles_data = AccelerationStructureGeometryTrianglesData {
        max_vertex: vertex_buffer.len() as _,
        vertex_data: Some(vertex_buffer.clone().into_bytes()),
        vertex_stride: size_of::<[i32; 3]>() as _,
        index_data: Some(IndexBuffer::U32(index_buffer)),
        ..AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT)
    };

    let geometries = AccelerationStructureGeometries::Triangles(vec![as_geometry_triangles_data]);

    unsafe {
        build_acceleration_structure_common(
            geometries,
            primitive_count,
            AccelerationStructureType::BottomLevel,
            memory_allocator,
            command_buffer_allocator,
            device,
            queue,
        )
    }
}

pub(crate) unsafe fn build_top_level_acceleration_structure(
    as_instances: Vec<AccelerationStructureInstance>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let primitive_count = as_instances.len() as u32;

    let instance_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        as_instances,
    )
    .unwrap();

    let as_geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
        AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer)),
    );

    let geometries = AccelerationStructureGeometries::Instances(as_geometry_instances_data);

    unsafe {
        build_acceleration_structure_common(
            geometries,
            primitive_count,
            AccelerationStructureType::TopLevel,
            memory_allocator,
            command_buffer_allocator,
            device,
            queue,
        )
    }
}
