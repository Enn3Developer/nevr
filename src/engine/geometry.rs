use crate::ToBytes;
use crate::engine::voxel::{RenderVoxelType, VoxelType};
use bevy::platform::collections::HashMap;
use bevy::prelude::{AssetId, Res, ResMut, Resource, Transform, Vec3};
use bevy::render::render_asset::ExtractedAssets;
use bevy::render::render_resource::{Buffer, BufferInitDescriptor, BufferUsages};
use bevy::render::renderer::RenderDevice;
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
        let mut vertices = Vec::with_capacity((VERTICES.len() + VERTICES.len() / 3) * voxels.len());
        let mut indices = Vec::with_capacity(INDICES.len() * voxels.len());
        let mut normals = Vec::with_capacity(NORMALS.len() * voxels.len());
        let mut offset = 0;

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
                vertices.push(1.0);
            }

            for index in INDICES {
                indices.push(index + offset * (VERTICES.len() as u32 / 3));
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
