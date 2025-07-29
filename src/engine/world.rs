use crate::context::{build_acceleration_structure_voxels, build_top_level_acceleration_structure};
use crate::voxel::{Voxel, VoxelLibrary, VoxelMaterial, VoxelType};
use crate::vulkan_instance::VulkanInstance;
use itertools::Itertools;
use std::sync::Arc;
use vulkano::Packed24_8;
use vulkano::acceleration_structure::{
    AabbPositions, AccelerationStructure, AccelerationStructureInstance,
};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

pub struct VoxelWorld {
    vulkan_instance: Arc<VulkanInstance>,
    voxel_library: VoxelLibrary,
    blas: Vec<Arc<AccelerationStructure>>,
    tlas: Vec<Arc<AccelerationStructure>>,
}

impl VoxelWorld {
    pub fn new(vulkan_instance: Arc<VulkanInstance>, voxel_library: VoxelLibrary) -> Self {
        Self {
            vulkan_instance,
            voxel_library,
            blas: vec![],
            tlas: vec![],
        }
    }

    pub fn update(
        &mut self,
        blocks: impl IntoIterator<Item = (u32, glm::Vec3)>,
    ) -> (Subbuffer<[VoxelMaterial]>, Subbuffer<[Voxel]>) {
        let block_chunks = blocks
            .into_iter()
            .filter_map(|(id, position)| self.voxel_library.create_block(id, position));

        // vec of voxels (Vec<Voxel>)
        let voxel_chunks = block_chunks
            .into_iter()
            .map(|block| block.voxel_array().into_iter().collect_vec())
            .flatten()
            .chunks(8192)
            .into_iter()
            .map(|v| v.collect_vec())
            .collect_vec();

        let material_data = Buffer::from_iter(
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
            self.voxel_library.materials.clone(),
        )
        .unwrap();

        let voxel_data = Buffer::from_iter(
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
            voxel_chunks.clone().into_iter().flatten().collect_vec(),
        )
        .unwrap();

        let voxel_buffers = voxel_chunks
            .into_iter()
            .map(|voxels| {
                Buffer::from_iter(
                    self.vulkan_instance.memory_allocator(),
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
                    self.vulkan_instance.memory_allocator(),
                    self.vulkan_instance.command_buffer_allocator(),
                    self.vulkan_instance.device(),
                    self.vulkan_instance.queue(),
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
                self.vulkan_instance.memory_allocator(),
                self.vulkan_instance.command_buffer_allocator(),
                self.vulkan_instance.device(),
                self.vulkan_instance.queue(),
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

    pub fn new_material(&mut self, id: u32, voxel_material: VoxelMaterial) {
        self.voxel_library.new_material(id, voxel_material);
    }

    pub fn new_type(&mut self, id: u32, voxel_type: VoxelType) {
        self.voxel_library.new_type(id, voxel_type);
    }
}
