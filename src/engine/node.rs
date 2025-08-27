use bevy::ecs::query::QueryItem;
use bevy::prelude::{FromWorld, World};
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode};
use bevy::render::renderer::RenderContext;
use bevy::render::view::ViewTarget;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct NEVRNodeLabel;

pub struct NEVRNode;

impl FromWorld for NEVRNode {
    fn from_world(_world: &mut World) -> Self {
        Self
    }
}

impl ViewNode for NEVRNode {
    type ViewQuery = (&'static ViewTarget);

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        _render_context: &mut RenderContext<'w>,
        _view_query: QueryItem<'w, '_, Self::ViewQuery>,
        _world: &'w World,
    ) -> Result<(), NodeRunError> {
        // TODO: bind the second group and dispatch the main compute pipeline
        // TODO: write the compute shader
        Ok(())
    }
}
