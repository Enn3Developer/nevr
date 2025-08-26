use crate::engine::voxel::{VoxelLibrary, VoxelTypeId};
use bevy::prelude::{Res, ResMut, Resource};
use bevy::render::render_resource::Blas;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use std::collections::VecDeque;

#[derive(Resource, Default)]
pub struct BlasManager {
    blas: Vec<Blas>,
    compaction_queue: VecDeque<(VoxelTypeId, u32, bool)>,
}

impl BlasManager {
    pub fn get(&self, id: VoxelTypeId) -> Option<&Blas> {
        self.blas.get(id as usize)
    }
}

pub fn prepare_blas(
    mut blas_manager: ResMut<BlasManager>,
    library: Res<VoxelLibrary>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
}
