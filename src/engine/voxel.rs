use crate::engine::color::{IntoRgba, VoxelColor};
use bevy::asset::AssetId;
use bevy::ecs::query::QueryItem;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::prelude::{Asset, Component, Handle, Transform, TypePath, Vec3, Visibility};
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_asset::{PrepareAssetError, RenderAsset};
use bevy::render::renderer::RenderDevice;
use bytemuck::{Pod, Zeroable};

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

#[derive(Asset, TypePath, Clone, Copy, Zeroable, Pod)]
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

    pub fn new_lambertian(diffuse: VoxelColor) -> Self {
        Self::new(diffuse.into_rgba(), 0.0, 0.0, MaterialModel::Lambertian)
    }

    pub fn new_metallic(diffuse: VoxelColor, fuzziness: f32) -> Self {
        Self::new(diffuse.into_rgba(), fuzziness, 0.0, MaterialModel::Metallic)
    }

    pub fn new_dielectric(diffuse: VoxelColor, refraction_index: f32) -> Self {
        Self::new(
            diffuse.into_rgba(),
            0.0,
            refraction_index,
            MaterialModel::Dielectric,
        )
    }

    pub fn new_diffuse_light(diffuse: VoxelColor, brightness: f32) -> Self {
        let mut diffuse = diffuse.into_rgba();
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
    pub fn new(min: Vec3, max: Vec3, material_id: u32) -> Self {
        Self {
            min: [min.x, min.y, min.z],
            max: [max.x, max.y, max.z],
            material_id,
            _padding_1: 0,
        }
    }
}

#[derive(Component, Debug)]
#[require(Transform, Visibility::Inherited)]
pub struct VoxelBlock {
    pub voxel_type: Handle<VoxelType>,
}

impl VoxelBlock {
    pub fn new(voxel_type: Handle<VoxelType>) -> Self {
        Self { voxel_type }
    }

    // TODO: move this code somewhere else
    // pub fn voxel_array(&self, transform: &GlobalTransform) -> impl IntoIterator<Item = Voxel> {
    //     let voxel_size = 1.0 / self.voxel_type.size as f32;
    //
    //     self.voxel_type
    //         .voxels
    //         .voxels
    //         .iter()
    //         .map(move |(material, pos)| {
    //             Voxel::new(
    //                 *transform * (pos * voxel_size),
    //                 *transform * ((pos + 1.0) * voxel_size),
    //                 *material,
    //             )
    //         })
    // }
}

#[derive(Component, Debug)]
pub struct RenderVoxelBlock {
    pub voxel_type: AssetId<VoxelType>,
}

impl ExtractComponent for VoxelBlock {
    type QueryData = (&'static VoxelBlock, &'static Transform);
    type QueryFilter = ();
    type Out = (RenderVoxelBlock, Transform);

    fn extract_component(
        (block, transform): QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some((
            RenderVoxelBlock {
                voxel_type: block.voxel_type.id(),
            },
            *transform,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct RelativeVoxel {
    pub material: Handle<VoxelMaterial>,
    pub position: Vec3,
}

impl RelativeVoxel {
    pub fn new(material: Handle<VoxelMaterial>, position: Vec3) -> Self {
        Self { material, position }
    }
}

#[derive(Asset, TypePath, Debug, Clone)]
pub struct VoxelType {
    size: i32,
    voxels: Vec<RelativeVoxel>,
}

impl VoxelType {
    pub fn new(size: u32, voxels: Vec<RelativeVoxel>) -> Self {
        Self {
            voxels,
            size: size as i32,
        }
    }

    pub fn size(&self) -> i32 {
        self.size
    }

    pub fn voxels(&self) -> &[RelativeVoxel] {
        &self.voxels
    }
}

#[derive(Asset, TypePath, Debug)]
pub struct RenderVoxelType;

impl RenderAsset for RenderVoxelType {
    type SourceAsset = VoxelType;
    type Param = SRes<RenderDevice>;

    fn prepare_asset(
        _source_asset: Self::SourceAsset,
        _asset_id: AssetId<Self::SourceAsset>,
        _render_device: &mut SystemParamItem<Self::Param>,
        _previous_asset: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        Ok(Self)
    }
}
