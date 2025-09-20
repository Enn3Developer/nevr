//! This module contains all the necessary structs to create blocks.

use crate::ToBytes;
use bevy::asset::AssetId;
use bevy::ecs::query::QueryItem;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::prelude::{
    Asset, Color, ColorToComponents, Component, GlobalTransform, Handle, InheritedVisibility,
    LinearRgba, Transform, TypePath, Vec3, Visibility,
};
use bevy::render::extract_component::ExtractComponent;
use bevy::render::render_asset::{PrepareAssetError, RenderAsset};
use bevy::render::render_resource::ShaderType;
use bevy::render::render_resource::encase::internal::{
    AlignmentValue, BufferMut, WriteInto, Writer,
};
use bevy::render::render_resource::encase::private::{Metadata, SizeValue};
use bevy::render::renderer::RenderDevice;

/// Describes the model to use for a material, used in [VoxelMaterial].
pub enum VoxelMaterialModel {
    /// A pure color material, can also be called "diffuse"; in raster this is similar to an albedo.
    Lambertian,
    /// A reflective material with a configurable fuzziness value.
    /// The higher the fuzziness the less precise the reflection is.
    Metallic,
    /// A water/glass-like material, it both reflects and refracts the light.
    /// Water has a refraction index of about 1.33, whilst glass has about 1.5.
    Dielectric,
    /// NOT USED YET
    Isotropic,
    /// An emissive material, could be used for torches, lamps, etc...
    ///
    /// The brightness can be thought as a multiplier of the color, for example if the color is pure white
    /// and the brightness is 10, then the diffuse value would be RGBA(10.0, 10.0, 10.0, 10.0).
    /// A convenient method is provided through [VoxelMaterial::new_diffuse_light] for which you provide a base
    /// color and the brightness separately.
    DiffuseLight,
}

impl From<VoxelMaterialModel> for u32 {
    fn from(value: VoxelMaterialModel) -> Self {
        match value {
            VoxelMaterialModel::Lambertian => 0,
            VoxelMaterialModel::Metallic => 1,
            VoxelMaterialModel::Dielectric => 2,
            VoxelMaterialModel::Isotropic => 3,
            VoxelMaterialModel::DiffuseLight => 4,
        }
    }
}

/// Describes a material a voxel has.
///
/// To use this, add it to through the [bevy::prelude::AssetServer]:
/// ```rs
/// let handle = asset_server.add(VoxelMaterial::new_lambertian(VoxelColor::RGBA(1.0, 1.0, 1.0, 1.0)));
/// ```
#[derive(Asset, TypePath, Clone, Copy)]
#[repr(C)]
pub struct VoxelMaterial {
    diffuse: LinearRgba,
    _diffuse_texture_id: i32,
    fuzziness: f32,
    refraction_index: f32,
    material_model: u32,
}

impl VoxelMaterial {
    pub fn new(
        diffuse: LinearRgba,
        fuzziness: f32,
        refraction_index: f32,
        material_model: VoxelMaterialModel,
    ) -> Self {
        Self {
            diffuse,
            fuzziness,
            refraction_index,
            material_model: material_model.into(),
            _diffuse_texture_id: -1,
        }
    }

    /// Creates a new lambertian material.
    ///
    /// Check [VoxelMaterialModel::Lambertian] for more information.
    pub fn new_lambertian(diffuse: Color) -> Self {
        Self::new(
            diffuse.to_linear(),
            0.0,
            0.0,
            VoxelMaterialModel::Lambertian,
        )
    }

    /// Creates a new metallic material.
    ///
    /// Check [VoxelMaterialModel::Metallic] for more information.
    pub fn new_metallic(diffuse: Color, fuzziness: f32) -> Self {
        Self::new(
            diffuse.to_linear(),
            fuzziness,
            0.0,
            VoxelMaterialModel::Metallic,
        )
    }

    /// Creates a new dielectric material.
    ///
    /// Check [VoxelMaterialModel::Dielectric] for more information.
    pub fn new_dielectric(diffuse: Color, refraction_index: f32) -> Self {
        Self::new(
            diffuse.to_linear(),
            0.0,
            refraction_index,
            VoxelMaterialModel::Dielectric,
        )
    }

    /// Creates a new emissive material.
    ///
    /// Check [VoxelMaterialModel::DiffuseLight] for more information.
    pub fn new_diffuse_light(diffuse: Color, brightness: f32) -> Self {
        let mut diffuse = diffuse.to_linear();
        diffuse.red *= brightness;
        diffuse.green *= brightness;
        diffuse.blue *= brightness;
        diffuse.alpha *= brightness;

        Self::new(diffuse, 0.0, 0.0, VoxelMaterialModel::DiffuseLight)
    }
}

impl RenderAsset for VoxelMaterial {
    type SourceAsset = Self;
    type Param = SRes<RenderDevice>;

    fn prepare_asset(
        source_asset: Self::SourceAsset,
        _asset_id: AssetId<Self::SourceAsset>,
        _param: &mut SystemParamItem<Self::Param>,
        _previous_asset: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        Ok(source_asset)
    }
}

impl ShaderType for VoxelMaterial {
    type ExtraMetadata = ();
    const METADATA: Metadata<Self::ExtraMetadata> = Metadata {
        alignment: AlignmentValue::new(16),
        has_uniform_min_alignment: false,
        min_size: SizeValue::new(32),
        is_pod: false,
        extra: (),
    };
}

impl WriteInto for VoxelMaterial {
    fn write_into<B>(&self, writer: &mut Writer<B>)
    where
        B: BufferMut,
    {
        writer.write_slice(self.diffuse.to_f32_array().to_bytes());
        writer.write_slice(&self._diffuse_texture_id.to_le_bytes());
        writer.write_slice(&self.fuzziness.to_le_bytes());
        writer.write_slice(&self.refraction_index.to_le_bytes());
        writer.write_slice(&self.material_model.to_le_bytes());
    }
}

// TODO: reimplement the voxel struct as a component to spawn singular voxels (useful for particles)
// #[derive(Clone, Copy, Zeroable, Pod)]
// #[repr(C)]
// pub struct Voxel {
//     min: [f32; 3],
//     _padding_1: u32,
//     max: [f32; 3],
//     material_id: u32,
// }
//
// impl Voxel {
//     pub fn new(min: Vec3, max: Vec3, material_id: u32) -> Self {
//         Self {
//             min: [min.x, min.y, min.z],
//             max: [max.x, max.y, max.z],
//             material_id,
//             _padding_1: 0,
//         }
//     }
// }

/// A component that describes a block in the world.
///
/// Check [VoxelType] for more information.
///
/// Use [Transform] to control the size, the rotation and the position of the block in the world:
/// ```rs
/// // Transform at position x = 1, y = 2, z = 3 with a scale of 2 (i.e. it is two times bigger in all axis)
/// let transform = Transform::from_xyz(1.0, 2.0, 3.0).with_scale(Vec3::new(2.0, 2.0, 2.0));
/// commands.spawn((VoxelBlock::new(handle_voxel_type), transform));
/// ```
///
/// **Note:** if you rotate the block, the bounding volume in the BLAS (used to accelerate ray intersections) is still
/// axis-aligned (i.e. it doesn't rotate) so if you have multiple rotated blocks one next to each other
/// you may suffer degraded performance.
#[derive(Component, Debug)]
#[require(Transform, Visibility::Inherited)]
pub struct VoxelBlock {
    /// The type of the block.
    pub voxel_type: Handle<VoxelType>,
}

impl VoxelBlock {
    pub fn new(voxel_type: Handle<VoxelType>) -> Self {
        Self { voxel_type }
    }
}

/// Used in the rendering phase to extract all needed [VoxelBlock]s.
#[derive(Component, Debug)]
pub struct RenderVoxelBlock {
    pub voxel_type: AssetId<VoxelType>,
}

impl ExtractComponent for VoxelBlock {
    type QueryData = (
        &'static VoxelBlock,
        &'static GlobalTransform,
        &'static InheritedVisibility,
    );
    type QueryFilter = ();
    type Out = (RenderVoxelBlock, GlobalTransform, InheritedVisibility);

    fn extract_component(
        (block, transform, visibility): QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some((
            RenderVoxelBlock {
                voxel_type: block.voxel_type.id(),
            },
            *transform,
            *visibility,
        ))
    }
}

/// A relative voxel in the type of the block.
///
/// You can think of this as a voxel in the block with the position relative to the block's position, for example
/// a position of (0.0, 1.0, 0.0) describes a voxel that is at x = 0, y = 1, z = 0 **inside** the block.
/// Check [VoxelType] for more information.
///
/// Every voxel has its own material, check [VoxelMaterial] and [VoxelMaterialModel] for more information.
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

/// Describes a type of block.
///
/// A VoxelType has a size as in how much large is the largest dimension of the block (x-axis, y-axis or z-axis).
/// It has a list of voxels that are relative to the position of the block, check [RelativeVoxel] for more information about it.
/// All the voxels inside this type will be scaled to the scale of a block, which by default is large 1x1x1 (as in basic units, you can think in meters if it's easier for you).
///
/// Add this asset to [bevy::prelude::AssetServer]:
/// ```rs
/// let voxels = vec![RelativeVoxel::new(material, Vec3::ZERO)];
/// let voxel_type = asset_server.add(VoxelType::new(1, voxels));
/// ```
///
/// In the example above, the size is `1` because the largest dimension (either the x-axis, y-axis or z-axis)
/// is large 1 unit, the position of the `RelativeVoxel` is (0.0, 0.0, 0.0) because it is at that coordinates **inside** the block.
/// This means that the `RelativeVoxel` is as large as the block and its position is the same as the block.
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

/// Used in the rendering phase to extracts all needed [VoxelType]s.
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
