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

/// Describes the model to use for a material, used in [VoxelMaterial].
pub enum MaterialModel {
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

/// Describes a material a voxel has.
///
/// To use this, add it to through the [bevy::prelude::AssetServer]:
/// ```rs
/// let handle = asset_server.add(VoxelMaterial::new_lambertian(VoxelColor::RGBA(1.0, 1.0, 1.0, 1.0)));
/// ```
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

    /// Creates a new lambertian material.
    ///
    /// Check [MaterialModel::Lambertian] for more information.
    pub fn new_lambertian(diffuse: VoxelColor) -> Self {
        Self::new(diffuse.into_rgba(), 0.0, 0.0, MaterialModel::Lambertian)
    }

    /// Creates a new metallic material.
    ///
    /// Check [MaterialModel::Metallic] for more information.
    pub fn new_metallic(diffuse: VoxelColor, fuzziness: f32) -> Self {
        Self::new(diffuse.into_rgba(), fuzziness, 0.0, MaterialModel::Metallic)
    }

    /// Creates a new dielectric material.
    ///
    /// Check [MaterialModel::Dielectric] for more information.
    pub fn new_dielectric(diffuse: VoxelColor, refraction_index: f32) -> Self {
        Self::new(
            diffuse.into_rgba(),
            0.0,
            refraction_index,
            MaterialModel::Dielectric,
        )
    }

    /// Creates a new emissive material.
    ///
    /// Check [MaterialModel::DiffuseLight] for more information.
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

/// A component that describes a block in the world.
///
/// Use [Transform] to control the size, the rotation and the position of the block in the world.
///
/// Note: if you rotate the block, the bounding volume in the BLAS (used to accelerate ray intersections) is still
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

/// A relative voxel in the type of the block.
///
/// You can think of this as a Voxel in the block with the position relative to the block, for example
/// a position of (0.0, 1.0, 0.0) describes a voxel that is at x = 0, y = 1, z = 0 **inside** the block.
/// Check [VoxelType] for more information.
///
/// Every voxel has its own material, check [VoxelMaterial] and [MaterialModel] for more information.
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
