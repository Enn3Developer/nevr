//! This module contains resources and systems used in the rendering phase.

use crate::ToBytes;
use crate::engine::voxel::{RenderVoxelType, VoxelMaterial, VoxelType};
use bevy::platform::collections::HashMap;
use bevy::prelude::{AssetId, Res, ResMut, Resource, Transform, Vec3};
use bevy::render::render_asset::ExtractedAssets;
use bevy::render::render_resource::encase::internal::{
    AlignmentValue, BufferMut, WriteInto, Writer,
};
use bevy::render::render_resource::encase::private::{Metadata, SizeValue};
use bevy::render::render_resource::{
    Buffer, BufferInitDescriptor, BufferUsages, BufferVec, ShaderSize, ShaderType,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bytemuck::{Pod, Zeroable};
use itertools::Itertools;

#[rustfmt::skip]
pub const VERTICES: [f32; 72] = [
    // LEFT
    1.0, 1.0, 1.0,
    1.0, 0.0, 1.0,
    1.0, 1.0, 0.0,
    1.0, 0.0, 0.0,

    // BOTTOM
    1.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 0.0, 0.0,

    // FORWARD
    1.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    1.0, 0.0, 1.0,
    0.0, 0.0, 1.0,

    // RIGHT
    0.0, 1.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 0.0,

    // TOP
    1.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    1.0, 1.0, 0.0,
    0.0, 1.0, 0.0,

    // BACKWARD
    1.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
];

#[rustfmt::skip]
pub const NORMALS: [f32; 72] = [
    // LEFT
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,

    // BOTTOM
    0.0, -1.0, 0.0,
    0.0, -1.0, 0.0,
    0.0, -1.0, 0.0,
    0.0, -1.0, 0.0,

    // FORWARD
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,

    // RIGHT
    -1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0,

    // TOP
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,

    // BACKWARD
    0.0, 0.0, -1.0,
    0.0, 0.0, -1.0,
    0.0, 0.0, -1.0,
    0.0, 0.0, -1.0,
];

#[rustfmt::skip]
pub const INDICES: [u32; 36] = [
    // LEFT
    2, 0, 1,
    2, 1, 3,

    // BOTTOM
    6, 4, 5,
    6, 5, 7,

    // FORWARD
    10, 8, 9,
    10, 9, 11,

    // RIGHT
    13, 12, 14,
    15, 13, 14,

    // TOP
    17, 16, 18,
    19, 17, 18,

    // BACKWARD
    21, 20, 22,
    23, 21, 22,
];

/// Struct used to store the indices used for a geometry in the shader
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct RenderObject {
    pub index: u32,
    pub material_id: u32,
}

impl ShaderType for RenderObject {
    type ExtraMetadata = ();
    const METADATA: Metadata<Self::ExtraMetadata> = Metadata {
        alignment: AlignmentValue::new(4),
        has_uniform_min_alignment: false,
        min_size: SizeValue::new(8),
        is_pod: false,
        extra: (),
    };
}

impl WriteInto for RenderObject {
    fn write_into<B>(&self, writer: &mut Writer<B>)
    where
        B: BufferMut,
    {
        writer.write_slice(&self.index.to_le_bytes());
        writer.write_slice(&self.material_id.to_le_bytes());
    }
}

impl ShaderSize for RenderObject {}

/// Manages the buffers for all voxels in the scene.
#[derive(Resource)]
pub struct GeometryManager {
    geometries_vertices: HashMap<AssetId<VoxelType>, Buffer>,
    geometries_indices: HashMap<AssetId<VoxelType>, Buffer>,

    added_types: Vec<AssetId<VoxelType>>,
    added_materials: Vec<AssetId<VoxelMaterial>>,

    vertices: BufferVec<f32>,
    indices: BufferVec<u32>,
    normals: BufferVec<f32>,
    materials: BufferVec<VoxelMaterial>,
    material_map: BufferVec<u32>,

    object_map: HashMap<AssetId<VoxelType>, u32>,
    index_map: Vec<u32>,
    material_index_map: Vec<u32>,
}

impl GeometryManager {
    pub fn get_geometry_vertices(&self, id: &AssetId<VoxelType>) -> Option<&Buffer> {
        self.geometries_vertices.get(id)
    }

    pub fn get_geometry_indices(&self, id: &AssetId<VoxelType>) -> Option<&Buffer> {
        self.geometries_indices.get(id)
    }

    pub fn vertices(&self) -> &BufferVec<f32> {
        &self.vertices
    }

    pub fn indices(&self) -> &BufferVec<u32> {
        &self.indices
    }

    pub fn normals(&self) -> &BufferVec<f32> {
        &self.normals
    }

    pub fn materials(&self) -> &BufferVec<VoxelMaterial> {
        &self.materials
    }

    pub fn material_map(&self) -> &BufferVec<u32> {
        &self.material_map
    }

    pub fn get_object_id(&self, id: &AssetId<VoxelType>) -> Option<u32> {
        // cheap copy to have a more ergonomic function usage
        self.object_map.get(id).cloned()
    }

    pub fn get_index(&self, object_id: u32) -> Option<u32> {
        self.index_map.get(object_id as usize).cloned()
    }

    pub fn get_index_material(&self, object_id: u32) -> Option<u32> {
        self.material_index_map.get(object_id as usize).cloned()
    }

    pub fn index_of_material(&self, id: &AssetId<VoxelMaterial>) -> Option<u32> {
        for (i, material_id) in self.added_materials.iter().enumerate() {
            if material_id == id {
                return Some(i as u32);
            }
        }

        None
    }
}

impl Default for GeometryManager {
    fn default() -> Self {
        Self {
            geometries_vertices: HashMap::default(),
            geometries_indices: HashMap::default(),

            added_types: vec![],
            added_materials: vec![],

            vertices: BufferVec::new(BufferUsages::STORAGE),
            indices: BufferVec::new(BufferUsages::STORAGE),
            normals: BufferVec::new(BufferUsages::STORAGE),
            materials: BufferVec::new(BufferUsages::STORAGE),
            material_map: BufferVec::new(BufferUsages::STORAGE),

            object_map: HashMap::default(),
            index_map: vec![],
            material_index_map: vec![],
        }
    }
}

// TODO: a refactor may soon be necessary
/// Extracts all necessary data to copy in buffers.
pub fn prepare_geometry(
    mut geometry_manager: ResMut<GeometryManager>,
    voxel_types: Res<ExtractedAssets<RenderVoxelType>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    for id in &voxel_types.removed {
        geometry_manager.geometries_vertices.remove(id);
        geometry_manager.geometries_indices.remove(id);
    }

    let mut new_additions = false;

    for (id, voxel_type) in &voxel_types.extracted {
        let size = 1.0 / voxel_type.size() as f32;
        let voxels = voxel_type.voxels();
        let mut vertices = Vec::with_capacity((VERTICES.len() + VERTICES.len() / 3) * voxels.len());
        let mut indices = Vec::with_capacity(INDICES.len() * voxels.len());
        let mut offset = 0;
        // divided by 4 because in the shader we use a vec4 for indices
        let global_offset = geometry_manager.indices.len() as u32 / 4;

        let added = geometry_manager.added_types.contains(id);
        new_additions |= !added;

        // if not still added, object_id will be used to reference the object's data
        // and global_offset is the first set of geometry's indices
        if !added {
            let object_id = geometry_manager.added_types.len() as u32;
            let material_offset = geometry_manager.material_map.len() as u32;

            geometry_manager.added_types.push(*id);
            geometry_manager.object_map.insert(*id, object_id);
            geometry_manager.index_map.push(global_offset);
            geometry_manager.material_index_map.push(material_offset);
        }

        for voxel in voxels {
            let position = voxel.position * size;
            let transform =
                Transform::from_scale(Vec3::new(size, size, size)).with_translation(position);

            let chunks = VERTICES.iter().chunks(3);

            for vec in chunks.into_iter() {
                let vec = vec.collect_array::<3>().unwrap();
                let vertex = transform * Vec3::new(*vec[0], *vec[1], *vec[2]);
                vertices.push(vertex.x);
                vertices.push(vertex.y);
                vertices.push(vertex.z);

                if !added {
                    geometry_manager.vertices.push(vertex.x);
                    geometry_manager.vertices.push(vertex.y);
                    geometry_manager.vertices.push(vertex.z);
                    geometry_manager.vertices.push(1.0);
                }
            }

            let chunks = INDICES.iter().chunks(3);

            for index in chunks.into_iter() {
                let indices_array = index.collect_array::<3>().unwrap();
                indices.push(indices_array[0] + offset * (VERTICES.len() as u32 / 3));
                indices.push(indices_array[1] + offset * (VERTICES.len() as u32 / 3));
                indices.push(indices_array[2] + offset * (VERTICES.len() as u32 / 3));

                if !added {
                    let material_id = geometry_manager
                        .index_of_material(&voxel.material.id())
                        .unwrap();

                    geometry_manager.material_map.push(material_id);

                    geometry_manager.indices.push(
                        indices_array[0] + offset * (VERTICES.len() as u32 / 3) + global_offset,
                    );
                    geometry_manager.indices.push(
                        indices_array[1] + offset * (VERTICES.len() as u32 / 3) + global_offset,
                    );
                    geometry_manager.indices.push(
                        indices_array[2] + offset * (VERTICES.len() as u32 / 3) + global_offset,
                    );
                    geometry_manager.indices.push(0);
                }
            }

            if !added {
                let chunks = NORMALS.iter().chunks(3);

                for normal in chunks.into_iter() {
                    let normal_array = normal.collect_array::<3>().unwrap();

                    geometry_manager.normals.push(*normal_array[0]);
                    geometry_manager.normals.push(*normal_array[1]);
                    geometry_manager.normals.push(*normal_array[2]);
                    geometry_manager.normals.push(1.0);
                }
            }

            offset += 1;
        }

        let vertices = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::BLAS_INPUT | BufferUsages::STORAGE | BufferUsages::VERTEX,
            contents: vertices.to_bytes(),
        });

        let indices = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::BLAS_INPUT | BufferUsages::STORAGE | BufferUsages::INDEX,
            contents: indices.to_bytes(),
        });

        if new_additions {
            geometry_manager
                .vertices
                .write_buffer(&render_device, &render_queue);
            geometry_manager
                .indices
                .write_buffer(&render_device, &render_queue);
            geometry_manager
                .normals
                .write_buffer(&render_device, &render_queue);
            geometry_manager
                .material_map
                .write_buffer(&render_device, &render_queue);
        }

        geometry_manager.geometries_vertices.insert(*id, vertices);
        geometry_manager.geometries_indices.insert(*id, indices);
    }
}

/// Prepare materials used for rendering
pub fn prepare_materials(
    mut geometry_manager: ResMut<GeometryManager>,
    materials: Res<ExtractedAssets<VoxelMaterial>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let mut new_additions = false;

    for (id, material) in &materials.extracted {
        let added = geometry_manager.added_materials.contains(id);
        new_additions |= !added;

        if !added {
            geometry_manager.added_materials.push(*id);
            geometry_manager.materials.push(*material);
        }
    }

    if new_additions {
        geometry_manager
            .materials
            .write_buffer(&render_device, &render_queue);
    }
}
