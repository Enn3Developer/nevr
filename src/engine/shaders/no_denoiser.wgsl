#import bevy_render::view::View

@group(0) @binding(0) var view_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var view_input: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(2) var<uniform> view: View;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.viewport.zw)) {
        return;
    }

    let original_color = textureLoad(view_input, global_id.xy);
    textureStore(view_output, global_id.xy, original_color);
}