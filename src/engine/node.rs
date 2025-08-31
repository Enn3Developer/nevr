//! This module contains the renderer code.

use crate::engine::camera::RayCamera;
use crate::engine::light::RenderVoxelLight;
use crate::{VoxelBindings, VoxelViewTarget};
use bevy::app::App;
use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::image::ToExtents;
use bevy::prelude::{AssetServer, FromWorld, Plugin, World};
use bevy::render::RenderApp;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::{
    BindGroupEntries, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
    DynamicUniformBuffer, PipelineCache,
};
use bevy::render::renderer::{RenderContext, RenderQueue};
use bevy::render::view::{ViewTarget, ViewUniformOffset, ViewUniforms};

pub struct NEVRNodeRender;

impl Plugin for NEVRNodeRender {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/raytracing.wgsl");
        embedded_asset!(app, "shaders/denoiser.wgsl");
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .add_render_graph_node::<ViewNodeRunner<NEVRNode>>(Core3d, NEVRNodeLabel)
            .add_render_graph_edges(
                Core3d,
                (Node3d::StartMainPass, NEVRNodeLabel, Node3d::MainOpaquePass),
            );
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct NEVRNodeLabel;

pub struct NEVRNode {
    pipeline: CachedComputePipelineId,
    denoise_pipeline: CachedComputePipelineId,
}

impl FromWorld for NEVRNode {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let voxel_bindings = world.resource::<VoxelBindings>();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("voxel_raytracing_pipeline".into()),
            layout: voxel_bindings.bind_group_layouts[..2].to_vec(),
            shader: load_embedded_asset!(world, "shaders/raytracing.wgsl"),
            ..Default::default()
        });

        let denoise_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("voxel_denoise_pipeline".into()),
            layout: voxel_bindings.bind_group_layouts[2..].to_vec(),
            shader: load_embedded_asset!(world, "shaders/denoiser.wgsl"),
            ..Default::default()
        });

        Self {
            pipeline,
            denoise_pipeline,
        }
    }
}

impl ViewNode for NEVRNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ExtractedCamera,
        &'static RayCamera,
        &'static ViewUniformOffset,
        &'static VoxelViewTarget,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_target, extracted_camera, camera, view_uniform_offset, voxel_view_target): QueryItem<
            'w,
            '_,
            Self::ViewQuery,
        >,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let voxel_bindings = world.resource::<VoxelBindings>();
        let render_queue = world.resource::<RenderQueue>();
        let view_uniforms = world.resource::<ViewUniforms>();
        let voxel_light = world.resource::<RenderVoxelLight>();

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(self.pipeline) else {
            eprintln!(
                "{:?}",
                pipeline_cache.get_compute_pipeline_state(self.pipeline)
            );
            return Ok(());
        };
        let Some(denoise_pipeline) = pipeline_cache.get_compute_pipeline(self.denoise_pipeline)
        else {
            eprintln!(
                "{:?}",
                pipeline_cache.get_compute_pipeline_state(self.denoise_pipeline)
            );
            return Ok(());
        };
        let Some(viewport) = &extracted_camera.physical_viewport_size else {
            eprintln!("no viewport size");
            return Ok(());
        };
        let Some(bind_group) = &voxel_bindings.bind_group else {
            eprintln!("no bind group");
            return Ok(());
        };
        let Some(view_uniforms) = view_uniforms.uniforms.binding() else {
            eprintln!("no view uniforms");
            return Ok(());
        };

        let mut camera_uniform = DynamicUniformBuffer::default();
        camera_uniform.push(camera);
        camera_uniform.write_buffer(render_context.render_device(), render_queue);
        let mut light_uniform = DynamicUniformBuffer::default();
        light_uniform.push(voxel_light);
        light_uniform.write_buffer(render_context.render_device(), render_queue);

        let camera_bind_group = render_context.render_device().create_bind_group(
            "voxel_bindings_camera",
            &voxel_bindings.bind_group_layouts[1],
            &BindGroupEntries::sequential((
                camera_uniform.binding().unwrap(),
                &voxel_view_target.0.default_view,
                light_uniform.binding().unwrap(),
                view_uniforms.clone(),
            )),
        );

        let denoise_bind_group = render_context.render_device().create_bind_group(
            "voxel_bindings_denoise",
            &voxel_bindings.bind_group_layouts[2],
            &BindGroupEntries::sequential((
                view_target.get_unsampled_color_attachment().view,
                &voxel_view_target.0.default_view,
                view_uniforms,
            )),
        );

        let command_encoder = render_context.command_encoder();

        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("voxel_raytracing"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_bind_group(1, &camera_bind_group, &[view_uniform_offset.offset]);
        pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);

        pass.set_pipeline(denoise_pipeline);
        pass.set_bind_group(0, &denoise_bind_group, &[view_uniform_offset.offset]);
        pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);

        Ok(())
    }
}
