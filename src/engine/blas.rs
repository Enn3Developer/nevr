//! This module contains the necessary resources and systems to manage BLASes (used to accelerate ray intersections).

use crate::engine::geometry::GeometryManager;
use crate::engine::voxel::{RenderVoxelType, VoxelType};
use bevy::mesh::VertexFormat;
use bevy::platform::collections::HashMap;
use bevy::prelude::{AssetId, Res, ResMut, Resource};
use bevy::render::render_asset::ExtractedAssets;
use bevy::render::render_resource::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, Blas, BlasBuildEntry, BlasGeometries,
    BlasGeometrySizeDescriptors, BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor,
    CommandEncoderDescriptor, CreateBlasDescriptor, IndexFormat,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use std::collections::VecDeque;

// Code based on bevy_solari's blas management

const MAX_COMPACTION_VERTICES_PER_FRAME: u32 = 400_000;

#[derive(Resource, Default)]
pub struct BlasManager {
    blas: HashMap<AssetId<VoxelType>, Blas>,
    compaction_queue: VecDeque<(AssetId<VoxelType>, u32, bool)>,
}

impl BlasManager {
    pub fn get(&self, id: &AssetId<VoxelType>) -> Option<&Blas> {
        self.blas.get(id)
    }
}

pub fn prepare_blas(
    mut blas_manager: ResMut<BlasManager>,
    geometry_manager: Res<GeometryManager>,
    voxel_types: Res<ExtractedAssets<RenderVoxelType>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    for id in &voxel_types.removed {
        blas_manager.blas.remove(id);
    }

    if voxel_types.extracted.is_empty() {
        return;
    }

    let blas_resources = voxel_types
        .extracted
        .iter()
        .map(|(id, _voxel_type)| {
            let vertices = geometry_manager.get_vertices(id).unwrap();
            let indices = geometry_manager.get_indices(id).unwrap();

            let (blas, blas_size) = allocate_blas(
                vertices.size() as u32,
                indices.size() as u32,
                &render_device,
            );
            blas_manager.blas.insert(*id, blas);
            blas_manager
                .compaction_queue
                .push_back((*id, blas_size.vertex_count, false));
            (*id, vertices, indices, blas_size)
        })
        .collect::<Vec<_>>();

    let build_entries = blas_resources
        .iter()
        .map(|(id, vertices, indices, blas_size)| {
            let geometry = BlasTriangleGeometry {
                size: blas_size,
                vertex_buffer: vertices,
                first_vertex: 0,
                vertex_stride: 16,
                index_buffer: Some(indices),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            };

            BlasBuildEntry {
                blas: &blas_manager.blas[id],
                geometry: BlasGeometries::TriangleGeometries(vec![geometry]),
            }
        })
        .collect::<Vec<_>>();

    let mut command_encoder =
        render_device.create_command_encoder(&CommandEncoderDescriptor::default());
    command_encoder.build_acceleration_structures(&build_entries, &[]);
    render_queue.submit([command_encoder.finish()]);
}

pub fn compact_blas(mut blas_manager: ResMut<BlasManager>, render_queue: Res<RenderQueue>) {
    let queue_size = blas_manager.compaction_queue.len();
    let mut blocks_processed = 0;
    let mut vertices_processed = 0;

    while !blas_manager.compaction_queue.is_empty()
        && vertices_processed < MAX_COMPACTION_VERTICES_PER_FRAME
        && blocks_processed < queue_size
    {
        blocks_processed += 1;
        let (id, count, processing) = blas_manager.compaction_queue.pop_front().unwrap();

        let Some(blas) = blas_manager.get(&id) else {
            continue;
        };

        if !processing {
            blas.prepare_compaction_async(|_| {});
        }

        if blas.ready_for_compaction() {
            let compacted_blas = render_queue.compact_blas(blas);
            blas_manager.blas.insert(id, compacted_blas);
            vertices_processed += count;
            continue;
        }

        blas_manager.compaction_queue.push_back((id, count, true));
    }
}

fn allocate_blas(
    vertices_size: u32,
    indices_size: u32,
    render_device: &RenderDevice,
) -> (Blas, BlasTriangleGeometrySizeDescriptor) {
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        // 4 floats in a vertex, 4 bytes in a float
        vertex_count: vertices_size / 4 / 16,
        index_format: Some(IndexFormat::Uint32),
        // 4 bytes per int
        index_count: Some(indices_size / 4),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = render_device.wgpu_device().create_blas(
        &CreateBlasDescriptor {
            label: None,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE
                | AccelerationStructureFlags::ALLOW_COMPACTION,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    (blas, blas_size)
}
