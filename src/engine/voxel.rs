use bevy::prelude::Resource;
use bytemuck::{Pod, Zeroable};
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

    pub fn new_lambertian(diffuse: [f32; 4]) -> Self {
        Self::new(diffuse, 0.0, 0.0, MaterialModel::Lambertian)
    }

    pub fn new_metallic(diffuse: [f32; 4], fuzziness: f32) -> Self {
        Self::new(diffuse, fuzziness, 0.0, MaterialModel::Metallic)
    }

    pub fn new_dielectric(diffuse: [f32; 4], refraction_index: f32) -> Self {
        Self::new(diffuse, 0.0, refraction_index, MaterialModel::Dielectric)
    }

    pub fn new_diffuse_light(mut diffuse: [f32; 4], brightness: f32) -> Self {
        diffuse[0] *= brightness;
        diffuse[1] *= brightness;
        diffuse[2] *= brightness;
        diffuse[3] *= brightness;

        Self::new(diffuse, 0.0, 0.0, MaterialModel::DiffuseLight)
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
    pub fn new(min: glm::Vec3, max: glm::Vec3, material_id: u32) -> Self {
        Self {
            min: [min.x, min.y, min.z],
            max: [max.x, max.y, max.z],
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
    min: glm::Vec3,
    voxel_type: Arc<VoxelType>,
}

impl VoxelBlock {
    pub fn new(min: glm::Vec3, voxel_type: Arc<VoxelType>) -> Self {
        Self { min, voxel_type }
    }

    pub fn voxel_array(&self) -> impl IntoIterator<Item = Voxel> {
        let voxel_size = 1.0 / self.voxel_type.size as f32;

        self.voxel_type
            .voxels
            .voxels
            .iter()
            .map(move |(material, pos)| {
                Voxel::new(
                    self.min + pos * voxel_size,
                    self.min + (pos.add_scalar(1.0)) * voxel_size,
                    *material,
                )
            })
    }
}

pub struct RelativeVoxel {
    voxels: Vec<(u32, glm::Vec3)>,
}

impl RelativeVoxel {
    pub fn new(voxels: impl IntoIterator<Item = (impl Into<u32>, glm::Vec3)>) -> Self {
        Self {
            voxels: voxels
                .into_iter()
                .map(|(id, pos)| (id.into(), pos))
                .collect(),
        }
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

#[derive(Resource)]
pub struct VoxelLibrary {
    voxels: Vec<Arc<VoxelType>>,
    pub(crate) materials: Vec<VoxelMaterial>,
}

impl VoxelLibrary {
    pub fn new() -> Self {
        Self {
            voxels: vec![],
            materials: vec![],
        }
    }

    pub fn new_material(&mut self, id: impl Into<u32>, voxel_material: VoxelMaterial) {
        let material_id = id.into();

        let mut difference = material_id as isize - self.materials.len() as isize;

        if difference < 0 {
            *self.materials.get_mut(material_id as usize).unwrap() = voxel_material;
            return;
        }

        while difference >= 0 {
            self.materials.push(voxel_material.clone());
            difference -= 1;
        }
    }

    pub fn new_type(&mut self, id: impl Into<u32>, voxel_type: VoxelType) {
        let voxel_type = Arc::new(voxel_type);
        let voxel_id = id.into();

        let mut difference = voxel_id as isize - self.voxels.len() as isize;

        if difference < 0 {
            *self.voxels.get_mut(voxel_id as usize).unwrap() = voxel_type;
            return;
        }

        while difference >= 0 {
            self.voxels.push(voxel_type.clone());
            difference -= 1;
        }
    }

    pub fn create_block(&self, id: impl Into<u32>, position: glm::Vec3) -> Option<VoxelBlock> {
        let voxel_type = self.voxels.get(id.into() as usize)?.clone();

        Some(VoxelBlock::new(position, voxel_type))
    }
}
