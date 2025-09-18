//! This module contains the renderer code.

use crate::engine::camera::RayCamera;
use crate::engine::light::RenderVoxelLight;
use crate::engine::skybox::VoxelSkybox;
use crate::{VoxelBindings, VoxelGBuffer, VoxelViewTarget};
use bevy::app::App;
use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::{FromWorld, Plugin, World};
use bevy::render::RenderApp;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::{
    BindGroupEntries, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
    DynamicUniformBuffer, PipelineCache,
};
use bevy::render::renderer::{RenderContext, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::view::{ViewUniformOffset, ViewUniforms};
use bevy::shader::ShaderDefVal;

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
                (Node3d::StartMainPass, NEVRNodeLabel, Node3d::MainOpaquePass),
            );
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct NEVRNodeLabel;

pub struct NEVRNode {
    pipeline: CachedComputePipelineId,
    skybox_pipeline: CachedComputePipelineId,
}

impl FromWorld for NEVRNode {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let voxel_bindings = world.resource::<VoxelBindings>();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("voxel_raytracing_pipeline".into()),
            layout: voxel_bindings.bind_group_layouts[..3].to_vec(),
            shader: load_embedded_asset!(world, "shaders/raytracing.wgsl"),
            ..Default::default()
        });

        let skybox_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("voxel_raytracing_pipeline".into()),
            layout: voxel_bindings.bind_group_layouts[..].to_vec(),
            shader: load_embedded_asset!(world, "shaders/raytracing.wgsl"),
            shader_defs: vec![ShaderDefVal::Bool("SKYBOX".into(), true)],
            ..Default::default()
        });

        Self {
            pipeline,
            skybox_pipeline,
        }
    }
}

impl ViewNode for NEVRNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static RayCamera,
        &'static ViewUniformOffset,
        &'static VoxelViewTarget,
        &'static VoxelGBuffer,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (extracted_camera, camera, view_uniform_offset, voxel_view_target, g_buffer): QueryItem<
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
        let optional_skybox = world.get_resource::<VoxelSkybox>();

        let pipeline_id = if optional_skybox.is_some() {
            self.skybox_pipeline
        } else {
            self.pipeline
        };

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
            eprintln!(
                "{:?}",
                pipeline_cache.get_compute_pipeline_state(pipeline_id)
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
                &voxel_view_target.output.default_view,
                light_uniform.binding().unwrap(),
                view_uniforms.clone(),
                &voxel_view_target.accumulation.default_view,
            )),
        );

        let g_buffer_bind_group = render_context.render_device().create_bind_group(
            "voxel_bindings_g_buffer",
            &voxel_bindings.bind_group_layouts[2],
            &BindGroupEntries::sequential((
                &g_buffer.albedo.default_view,
                &g_buffer.normal.default_view,
                &g_buffer.world_position.default_view,
            )),
        );

        let optional_skybox_bind_group = if let Some(skybox) = optional_skybox {
            let gpu_images = world.resource::<RenderAssets<GpuImage>>();
            let Some(image) = gpu_images.get(skybox.0.id()) else {
                eprintln!("no skybox image found");
                return Ok(());
            };

            Some(render_context.render_device().create_bind_group(
                "voxel_bindings_skybox",
                &voxel_bindings.bind_group_layouts[3],
                &BindGroupEntries::sequential((&image.texture_view, &image.sampler)),
            ))
        } else {
            None
        };

        let command_encoder = render_context.command_encoder();

        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("voxel_raytracing"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_bind_group(1, &camera_bind_group, &[view_uniform_offset.offset]);
        pass.set_bind_group(2, &g_buffer_bind_group, &[]);
        if let Some(skybox_bind_group) = optional_skybox_bind_group.as_ref() {
            pass.set_bind_group(3, skybox_bind_group, &[]);
        }
        pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);

        Ok(())
    }
}
