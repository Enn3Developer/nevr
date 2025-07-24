use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::acceleration_structure::AabbPositions;

pub enum MaterialModel {
    Lambertian,
    Metallic,
    Dielectric,
    Isotropic,
    DiffuseLight,
}

impl From<MaterialModel> for u32 {
    fn from(value: MaterialModel) -> Self {
        match value {
            MaterialModel::Lambertian => 0,
            MaterialModel::Metallic => 1,
            MaterialModel::Dielectric => 2,
            MaterialModel::Isotropic => 3,
            MaterialModel::DiffuseLight => 4,
        }
    }
}

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct VoxelMaterial {
    diffuse: [f32; 4],
    _diffuse_texture_id: i32,
    fuzziness: f32,
    refraction_index: f32,
    material_model: u32,
}

impl VoxelMaterial {
    pub fn new(
        diffuse: [f32; 4],
        fuzziness: f32,
        refraction_index: f32,
        material_model: MaterialModel,
    ) -> Self {
        Self {
            diffuse,
            fuzziness,
            refraction_index,
            material_model: material_model.into(),
            _diffuse_texture_id: -1,
        }
    }
}

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Voxel {
    min: [f32; 3],
    _padding_1: u32,
    max: [f32; 3],
    material_id: u32,
}

impl Voxel {
    pub fn new(min: Vec3, max: Vec3, material_id: u32) -> Self {
        Self {
            min: min.to_array(),
            max: max.to_array(),
            material_id,
            _padding_1: 0,
        }
    }
}

impl From<Voxel> for AabbPositions {
    fn from(voxel: Voxel) -> Self {
        AabbPositions {
            min: voxel.min,
            max: voxel.max,
        }
    }
}

pub struct VoxelBlock {
    min: Vec3,
    voxel_type: Arc<VoxelType>,
}

impl VoxelBlock {
    pub fn new(min: Vec3, voxel_type: Arc<VoxelType>) -> Self {
        Self { min, voxel_type }
    }

    pub fn voxel_array(&self, voxel_library: &VoxelLibrary) -> impl IntoIterator<Item = Voxel> {
        let voxel_size = 1.0 / self.voxel_type.size as f32;

        self.voxel_type
            .voxels
            .voxels
            .iter()
            .map(|(material, pos)| (voxel_library.material_map.get(material).unwrap(), pos))
            .map(move |(material, pos)| {
                Voxel::new(
                    self.min + pos * voxel_size,
                    self.min + (pos + 1.0) * voxel_size,
                    *material,
                )
            })
    }
}

pub struct RelativeVoxel {
    voxels: Vec<(String, Vec3)>,
}

impl RelativeVoxel {
    pub fn new(voxels: Vec<(String, Vec3)>) -> Self {
        Self { voxels }
    }
}

pub struct VoxelType {
    size: i32,
    voxels: RelativeVoxel,
}

impl VoxelType {
    pub fn new(size: u32, voxels: RelativeVoxel) -> Self {
        Self {
            voxels,
            size: size as i32,
        }
    }
}

pub struct VoxelLibrary {
    voxel_map: HashMap<String, Arc<VoxelType>>,
    material_map: HashMap<String, u32>,
    pub(crate) materials: Vec<VoxelMaterial>,
}

impl VoxelLibrary {
    pub fn new() -> Self {
        Self {
            voxel_map: HashMap::new(),
            materials: vec![],
            material_map: HashMap::new(),
        }
    }

    pub fn new_material(&mut self, id: impl Into<String>, voxel_material: VoxelMaterial) {
        let material_id = self.materials.len();
        self.materials.push(voxel_material);

        self.material_map.insert(id.into(), material_id as u32);
    }

    pub fn new_type(&mut self, id: impl Into<String>, voxel_type: VoxelType) {
        let voxel_type = Arc::new(voxel_type);

        self.voxel_map.insert(id.into(), voxel_type);
    }

    pub fn create_block(&self, id: impl AsRef<str>, position: Vec3) -> Option<VoxelBlock> {
        let voxel_type = self.voxel_map.get(id.as_ref())?.clone();

        Some(VoxelBlock::new(position, voxel_type))
    }
}
