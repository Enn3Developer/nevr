//! This module contains resources and systems used in the rendering phase.

use crate::ToBytes;
use crate::engine::voxel::{RenderVoxelType, VoxelType};
use bevy::platform::collections::HashMap;
use bevy::prelude::{AssetId, Res, ResMut, Resource, Transform, Vec3};
use bevy::render::render_asset::ExtractedAssets;
use bevy::render::render_resource::{Buffer, BufferInitDescriptor, BufferUsages, BufferVec};
use bevy::render::renderer::{RenderDevice, RenderQueue};
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

/// Manages the buffers for all voxels in the scene.
#[derive(Resource)]
pub struct GeometryManager {
    geometries_vertices: HashMap<AssetId<VoxelType>, Buffer>,
    geometries_indices: HashMap<AssetId<VoxelType>, Buffer>,

    added_types: Vec<AssetId<VoxelType>>,
    vertices: BufferVec<f32>,
    indices: BufferVec<u32>,
    normals: BufferVec<f32>,
    object_map: HashMap<AssetId<VoxelType>, u32>,
    index_map: Vec<u32>,
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

    pub fn get_object_id(&self, id: &AssetId<VoxelType>) -> Option<u32> {
        // cheap copy to have a more ergonomic function usage
        self.object_map.get(id).cloned()
    }

    pub fn get_index(&self, object_id: u32) -> Option<u32> {
        self.index_map.get(object_id as usize).cloned()
    }
}

impl Default for GeometryManager {
    fn default() -> Self {
        Self {
            geometries_vertices: HashMap::default(),
            geometries_indices: HashMap::default(),
            added_types: vec![],
            vertices: BufferVec::new(BufferUsages::STORAGE),
            indices: BufferVec::new(BufferUsages::STORAGE),
            normals: BufferVec::new(BufferUsages::STORAGE),
            object_map: HashMap::default(),
            index_map: vec![],
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
        let object_id = geometry_manager.added_types.len() as u32;
        if !added {
            geometry_manager.added_types.push(*id);
            geometry_manager.object_map.insert(*id, object_id);
            geometry_manager.index_map.push(global_offset);
        }

        for voxel in voxels {
            let position = voxel.position * size;
            let transform =
                Transform::from_scale(Vec3::new(size, size, size)).with_translation(position);

            let chunks = VERTICES.iter().chunks(3);

            for vec in chunks.into_iter() {
                let vec: [&f32; 3] = vec.collect_array().unwrap();
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
        }

        geometry_manager.geometries_vertices.insert(*id, vertices);
        geometry_manager.geometries_indices.insert(*id, indices);
    }
}
