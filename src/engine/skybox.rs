//! Skybox module.

use bevy::prelude::{Handle, Image, Resource};
use bevy::render::extract_resource::ExtractResource;

/// Skybox resource.
///
/// The image provided **must** be a cubemap (DDS or KTX2, recommended DDS).
/// An easy way to create a DDS cubemap is to use a panorama image, convert it to 6 images (one for each face)
/// and use GIMP to export those images as a DDS cubemap.
/// For GIMP, import the images as layers and rename them as `positive x`, `negative x`, `positive y` and so on.
#[derive(Resource, ExtractResource, Clone, Debug)]
pub struct VoxelSkybox(pub Handle<Image>);
