use crate::VoxelBindings;
use bevy::app::App;
use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::{FromWorld, Plugin, World};
use bevy::render::RenderApp;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::{
    CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
};
use bevy::render::renderer::RenderContext;
use bevy::render::view::ViewTarget;

pub struct NEVRNodeRender;

impl Plugin for NEVRNodeRender {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/raytracing.wgsl");
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .add_render_graph_node::<ViewNodeRunner<NEVRNode>>(Core3d, NEVRNodeLabel)
            .add_render_graph_edges(
                Core3d,
                (Node3d::EndPrepasses, NEVRNodeLabel, Node3d::EndMainPass),
            );
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct NEVRNodeLabel;

pub struct NEVRNode {
    pipeline: CachedComputePipelineId,
}

impl FromWorld for NEVRNode {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let voxel_bindings = world.resource::<VoxelBindings>();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("voxel_raytracing_pipeline".into()),
            layout: voxel_bindings.bind_group_layouts.to_vec(),
            shader: load_embedded_asset!(world, "shaders/raytracing.wgsl"),
            ..Default::default()
        });

        Self { pipeline }
    }
}

impl ViewNode for NEVRNode {
    // TODO: find a way to extract RayCamera together with ViewTarget
    type ViewQuery = (&'static ViewTarget, &'static ExtractedCamera);

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_target, extracted_camera): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let voxel_bindings = world.resource::<VoxelBindings>();

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(self.pipeline) else {
            eprintln!(
                "{:?}",
                pipeline_cache.get_compute_pipeline_state(self.pipeline)
            );
            return Ok(());
        };
        let Some(viewport) = &extracted_camera.physical_viewport_size else {
            return Ok(());
        };

        // let command_encoder = render_context.command_encoder();
        //
        // let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        //     label: Some("voxel_raytracing"),
        //     timestamp_writes: None,
        // });
        //
        // // TODO: bind the BindGroups
        // pass.set_pipeline(pipeline);
        // pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);

        // println!("test");
        Ok(())
    }
}
