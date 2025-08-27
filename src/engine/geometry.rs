use crate::ToBytes;
use crate::engine::voxel::{RenderVoxelType, VoxelType};
use bevy::platform::collections::HashMap;
use bevy::prelude::{AssetId, Res, ResMut, Resource, Transform, Vec3, Vec4};
use bevy::render::render_asset::ExtractedAssets;
use bevy::render::render_resource::{Buffer, BufferInitDescriptor, BufferUsages};
use bevy::render::renderer::RenderDevice;
use itertools::Itertools;

// source for the data: https://raw.githubusercontent.com/McNopper/GLUS/master/GLUS/src/glus_shape.c
pub const VERTICES: [f32; 96] = [
    -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0,
    1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
    -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
    -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
];

pub const NORMALS: [f32; 72] = [
    0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
];

pub const INDICES: [u32; 36] = [
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 15, 14, 12, 14, 13, 16, 17, 18,
    16, 18, 19, 20, 23, 22, 20, 22, 21,
];

#[derive(Resource, Default)]
pub struct GeometryManager {
    geometries_vertices: HashMap<AssetId<VoxelType>, Buffer>,
    geometries_indices: HashMap<AssetId<VoxelType>, Buffer>,
    geometries_normals: HashMap<AssetId<VoxelType>, Buffer>,
}

impl GeometryManager {
    pub fn get_vertices(&self, id: &AssetId<VoxelType>) -> Option<&Buffer> {
        self.geometries_vertices.get(id)
    }

    pub fn get_indices(&self, id: &AssetId<VoxelType>) -> Option<&Buffer> {
        self.geometries_indices.get(id)
    }

    pub fn get_normals(&self, id: &AssetId<VoxelType>) -> Option<&Buffer> {
        self.geometries_normals.get(id)
    }
}

pub fn prepare_geometry(
    mut geometry_manager: ResMut<GeometryManager>,
    voxel_types: Res<ExtractedAssets<RenderVoxelType>>,
    render_device: Res<RenderDevice>,
) {
    for id in &voxel_types.removed {
        geometry_manager.geometries_vertices.remove(id);
        geometry_manager.geometries_indices.remove(id);
    }

    for (id, voxel_type) in &voxel_types.extracted {
        let size = 1.0 / voxel_type.size() as f32;
        let voxels = voxel_type.voxels();
        let mut vertices = Vec::with_capacity(VERTICES.len() * voxels.len());
        let mut indices = Vec::with_capacity(INDICES.len() * voxels.len());
        let mut normals = Vec::with_capacity(NORMALS.len() * voxels.len());
        let mut offset = 0;

        for (_material, position) in voxels {
            let position = position * size;
            let transform =
                Transform::from_scale(Vec3::new(size, size, size)).with_translation(position);

            let chunks = VERTICES.iter().chunks(4);

            for vec in chunks.into_iter() {
                let vec: [&f32; 4] = vec.collect_array().unwrap();
                let vertex = transform.to_matrix() * Vec4::new(*vec[0], *vec[1], *vec[2], *vec[3]);
                vertices.push(vertex.x);
                vertices.push(vertex.y);
                vertices.push(vertex.z);
                vertices.push(vertex.w);
            }

            for index in INDICES {
                indices.push(index + offset * INDICES.len() as u32);
            }

            for normal in NORMALS {
                normals.push(normal);
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

        let normals = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::STORAGE,
            contents: normals.to_bytes(),
        });

        geometry_manager.geometries_vertices.insert(*id, vertices);
        geometry_manager.geometries_indices.insert(*id, indices);
        geometry_manager.geometries_normals.insert(*id, normals);
    }
}
