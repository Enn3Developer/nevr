//! Denoiser module.

use crate::engine::node::NEVRNodeLabel;
use crate::{VoxelBindings, VoxelViewTarget};
use bevy::app::App;
use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::{FromWorld, Plugin, Resource, World};
use bevy::render::RenderApp;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::{
    BindGroupEntries, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
    PipelineCache,
};
use bevy::render::renderer::RenderContext;
use bevy::render::view::{ViewTarget, ViewUniformOffset, ViewUniforms};

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct DenoiserLabel;

/// Describes the denoiser to use for the rendering pipeline.
///
/// Quick summary:
/// - None: No denoiser.
/// - Simple: The simplest and fastest denoiser, worst quality.
///
/// Defaults to [VoxelDenoiser::None].
#[derive(Resource, ExtractResource, Clone, Copy, Debug, Default)]
pub enum VoxelDenoiser {
    /// Don't enable the denoiser pass
    #[default]
    None,
    /// The simplest denoiser, it's really fast but has the worst quality.
    Simple,
}

/// The plugin which adds a denoiser for the rendered image.
///
/// This is enabled by default when using [nevr::NEVRPlugin].
pub struct DenoiserPlugin;

impl Plugin for DenoiserPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/no_denoiser.wgsl");
        embedded_asset!(app, "shaders/simple_denoiser.wgsl");

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
    none_pipeline: CachedComputePipelineId,
    simple_pipeline: CachedComputePipelineId,
}

impl FromWorld for DenoiserNode {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let voxel_bindings = world.resource::<VoxelBindings>();
        let binding_layout = voxel_bindings.bind_group_layouts[2].clone();

        let none_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("voxel_none_denoiser_pipeline".into()),
            layout: vec![binding_layout.clone()],
            shader: load_embedded_asset!(world, "shaders/no_denoiser.wgsl"),
            ..Default::default()
        });

        let simple_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("voxel_simple_denoiser_pipeline".into()),
            layout: vec![binding_layout.clone()],
            shader: load_embedded_asset!(world, "shaders/simple_denoiser.wgsl"),
            ..Default::default()
        });

        Self {
            none_pipeline,
            simple_pipeline,
        }
    }
}

impl ViewNode for DenoiserNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ExtractedCamera,
        &'static ViewUniformOffset,
        &'static VoxelViewTarget,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_target, camera, view_uniform_offset, voxel_view_target): QueryItem<
            'w,
            '_,
            Self::ViewQuery,
        >,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let voxel_denoiser = world.resource::<VoxelDenoiser>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let voxel_bindings = world.resource::<VoxelBindings>();
        let view_uniforms = world.resource::<ViewUniforms>();

        let pipeline_id = match *voxel_denoiser {
            VoxelDenoiser::None => self.none_pipeline,
            VoxelDenoiser::Simple => self.simple_pipeline,
        };

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
            eprintln!(
                "{:?}",
                pipeline_cache.get_compute_pipeline_state(pipeline_id)
            );
            return Ok(());
        };

        let Some(viewport) = &camera.physical_viewport_size else {
            eprintln!("no viewport size");
            return Ok(());
        };

        let Some(view_uniforms) = view_uniforms.uniforms.binding() else {
            eprintln!("no view uniforms");
            return Ok(());
        };

        let denoise_bind_group = render_context.render_device().create_bind_group(
            "voxel_bindings_denoiser",
            &voxel_bindings.bind_group_layouts[2],
            &BindGroupEntries::sequential((
                view_target.get_unsampled_color_attachment().view,
                &voxel_view_target.0.default_view,
                view_uniforms,
            )),
        );

        let command_encoder = render_context.command_encoder();

        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("voxel_raytracing_denoiser"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &denoise_bind_group, &[view_uniform_offset.offset]);
        pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);

        Ok(())
    }
}
