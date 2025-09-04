//! Denoiser module.

use crate::engine::node::NEVRNodeLabel;
use crate::{VoxelBindings, VoxelGBuffer, VoxelViewTarget};
use bevy::app::App;
use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::image::ToExtents;
use bevy::prelude::{FromWorld, Plugin, Resource, UVec2, World};
use bevy::render::RenderApp;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{texture_storage_2d, uniform_buffer};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BindingResource,
    CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, IntoBinding,
    PipelineCache, ShaderStages, StorageTextureAccess, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureView, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::TextureCache;
use bevy::render::view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms};
use std::num::NonZeroU32;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct DenoiserLabel;

// TODO: refactor the various denoisers as a trait and implement the denoise pipeline using a resource that holds that dynamic object
/// Describes the denoiser to use for the rendering pipeline. It is recommended to try the various denoiser for
/// your particular scene.
///
/// Quick summary:
/// - None: No denoiser.
/// - Simple: The simplest and fastest denoiser, decent quality.
/// - ATrous: A bit more sophisticated, fast, good quality
///
/// Defaults to [VoxelDenoiser::None].
///
/// **Note:** By changing the samples count in [crate::engine::camera::VoxelCamera] the resulted denoised
/// image may vary by a lot.
#[derive(Resource, ExtractResource, Clone, Copy, Debug, Default)]
pub enum VoxelDenoiser {
    /// Doesn't enable the denoiser pass.
    #[default]
    None,
    /// The simplest denoiser, it's really fast but has the worst quality, for a better quality you have to increase the sample count.
    Simple,
    /// Implements the Edge-Avoiding À-Trous Wavelet denoiser based on [Dammertz et al. 2010](https://jo.dreggn.org/home/2010_atrous.pdf).
    ///
    /// Good image quality, and it's a fast denoiser.
    ///
    /// Params:
    /// - filter_size: how big should be the largest filter.
    ATrous(NonZeroU32),
}

/// The plugin which adds a denoiser for the rendered image.
///
/// This is enabled by default when using [nevr::NEVRPlugin].
pub struct DenoiserPlugin;

impl Plugin for DenoiserPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/simple_denoiser.wgsl");
        embedded_asset!(app, "shaders/a_trous.wgsl");

        app.add_plugins(ExtractResourcePlugin::<VoxelDenoiser>::default())
            .init_resource::<VoxelDenoiser>();
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .add_render_graph_node::<ViewNodeRunner<DenoiserNode>>(Core3d, DenoiserLabel)
            .add_render_graph_edges(
                Core3d,
                (NEVRNodeLabel, DenoiserLabel, Node3d::MainOpaquePass),
            );
    }
}

pub struct DenoiserNode {
    simple_pipeline: CachedComputePipelineId,
    simple_binding_layout: BindGroupLayout,

    a_trous_pipeline: CachedComputePipelineId,
    a_trous_binding_layouts: [BindGroupLayout; 2],
}

impl DenoiserNode {
    fn none_pipeline(
        &self,
        render_context: &mut RenderContext,
        view_output: &TextureView,
        view_input: &TextureView,
    ) {
        let command_encoder = render_context.command_encoder();
        command_encoder.copy_texture_to_texture(
            view_input.texture().as_image_copy(),
            view_output.texture().as_image_copy(),
            view_output.texture().size(),
        );
    }

    fn simple_pipeline(
        &self,
        render_context: &mut RenderContext,
        pipeline_cache: &PipelineCache,
        view_output: &TextureView,
        view_input: &TextureView,
        view_uniforms: BindingResource,
        view_uniform_offset: u32,
        viewport: &UVec2,
    ) {
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(self.simple_pipeline) else {
            eprintln!(
                "{:?}",
                pipeline_cache.get_compute_pipeline_state(self.simple_pipeline)
            );
            return;
        };

        let denoise_bind_group = render_context.render_device().create_bind_group(
            "voxel_bindings_simple_denoiser",
            &self.simple_binding_layout,
            &BindGroupEntries::sequential((view_output, view_input, view_uniforms)),
        );

        let command_encoder = render_context.command_encoder();

        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("voxel_raytracing_simple_denoiser"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &denoise_bind_group, &[view_uniform_offset]);
        pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);
    }

    fn a_trous_pipeline(
        &self,
        render_context: &mut RenderContext,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        pipeline_cache: &PipelineCache,
        view_output: &TextureView,
        view_input: &TextureView,
        view_uniforms: BindingResource,
        view_uniform_offset: u32,
        viewport: &UVec2,
        g_buffer: &VoxelGBuffer,
        size: u32,
    ) {
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(self.a_trous_pipeline) else {
            eprintln!(
                "{:?}",
                pipeline_cache.get_compute_pipeline_state(self.a_trous_pipeline)
            );
            return;
        };

        let denoise_bind_group = render_context.render_device().create_bind_group(
            "voxel_bindings_a_trous_denoiser",
            &self.a_trous_binding_layouts[0],
            &BindGroupEntries::sequential((
                view_uniforms,
                &g_buffer.albedo.default_view,
                &g_buffer.normal.default_view,
                &g_buffer.world_position.default_view,
            )),
        );

        let command_encoder = render_context.command_encoder();

        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("voxel_raytracing_a_trous_denoiser"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &denoise_bind_group, &[view_uniform_offset]);

        let mut i = 1;
        let mut index = 0;

        loop {
            if i > size {
                break;
            }

            let mut filter_uniform = UniformBuffer::default();
            *filter_uniform.get_mut() = i;
            filter_uniform.write_buffer(render_device, render_queue);

            let input = if index == 0 {
                view_input
            } else {
                &g_buffer.secondary_textures[index - 1].default_view
            };

            let filter_denoise_bind_group = render_device.create_bind_group(
                "voxel_bindings_a_trous_filter_denoiser",
                &self.a_trous_binding_layouts[1],
                &BindGroupEntries::sequential((
                    filter_uniform.binding().unwrap(),
                    &g_buffer.secondary_textures[index].default_view,
                    input,
                )),
            );

            pass.set_bind_group(1, &filter_denoise_bind_group, &[]);
            pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);

            i *= 2;
            index += 1;
        }

        drop(pass);

        command_encoder.copy_texture_to_texture(
            g_buffer
                .secondary_textures
                .last()
                .unwrap()
                .texture
                .as_image_copy(),
            view_output.texture().as_image_copy(),
            view_output.texture().size(),
        );
    }
}

impl FromWorld for DenoiserNode {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let simple_binding_layout = render_device.create_bind_group_layout(
            "voxel_simple_denoiser_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // View output
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                    // View input
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                    // View
                    uniform_buffer::<ViewUniform>(true),
                ),
            ),
        );

        let a_trous_binding_layout = render_device.create_bind_group_layout(
            "voxel_a_trous_denoiser_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // View
                    uniform_buffer::<ViewUniform>(true),
                    // Albedo
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                    // Normal
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                    // World position
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                ),
            ),
        );

        let a_trous_filter_a_trous_binding_layout = render_device.create_bind_group_layout(
            "voxel_a_trous_filter_denoiser_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // Filter size
                    uniform_buffer::<u32>(false),
                    // View output
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                    // View input
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadOnly),
                ),
            ),
        );

        let simple_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("voxel_simple_denoiser_pipeline".into()),
            layout: vec![simple_binding_layout.clone()],
            shader: load_embedded_asset!(world, "shaders/simple_denoiser.wgsl"),
            ..Default::default()
        });

        let a_trous_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("voxel_a_trous_denoiser_pipeline".into()),
            layout: vec![
                a_trous_binding_layout.clone(),
                a_trous_filter_a_trous_binding_layout.clone(),
            ],
            shader: load_embedded_asset!(world, "shaders/a_trous.wgsl"),
            ..Default::default()
        });

        Self {
            simple_pipeline,
            simple_binding_layout,

            a_trous_pipeline,
            a_trous_binding_layouts: [
                a_trous_binding_layout,
                a_trous_filter_a_trous_binding_layout,
            ],
        }
    }
}

impl ViewNode for DenoiserNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ExtractedCamera,
        &'static ViewUniformOffset,
        &'static VoxelViewTarget,
        &'static VoxelGBuffer,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_target, camera, view_uniform_offset, voxel_view_target, g_buffer): QueryItem<
            'w,
            '_,
            Self::ViewQuery,
        >,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();
        let voxel_denoiser = world.resource::<VoxelDenoiser>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let view_uniforms = world.resource::<ViewUniforms>();

        let Some(viewport) = &camera.physical_viewport_size else {
            eprintln!("no viewport size");
            return Ok(());
        };

        let Some(view_uniforms) = view_uniforms.uniforms.binding() else {
            eprintln!("no view uniforms");
            return Ok(());
        };

        match voxel_denoiser {
            VoxelDenoiser::None => self.none_pipeline(
                render_context,
                &TextureView::from(view_target.get_unsampled_color_attachment().view.clone()),
                &voxel_view_target.0.default_view,
            ),
            VoxelDenoiser::Simple => self.simple_pipeline(
                render_context,
                pipeline_cache,
                &TextureView::from(view_target.get_unsampled_color_attachment().view.clone()),
                &voxel_view_target.0.default_view,
                view_uniforms,
                view_uniform_offset.offset,
                viewport,
            ),
            VoxelDenoiser::ATrous(size) => self.a_trous_pipeline(
                render_context,
                render_device,
                render_queue,
                pipeline_cache,
                &TextureView::from(view_target.get_unsampled_color_attachment().view.clone()),
                &voxel_view_target.0.default_view,
                view_uniforms,
                view_uniform_offset.offset,
                viewport,
                &g_buffer,
                size.get(),
            ),
        }

        Ok(())
    }
}
