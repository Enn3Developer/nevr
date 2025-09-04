#import bevy_render::view::View

// https://jo.dreggn.org/home/2010_atrous.pdf

const COLOR_WEIGHT: f32 = 0.85;
const ALBEDO_WEIGHT: f32 = 0.4;
const NORMAL_WEIGHT: f32 = 0.3;
const WORLD_POSITION_WEIGHT: f32 = 0.25;

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var albedo_texture: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var normal_texture: texture_storage_2d<rgba16float, read>;
@group(0) @binding(3) var world_position_texture: texture_storage_2d<rgba16float, read>;

@group(1) @binding(0) var<uniform> step_width: u32;
@group(1) @binding(1) var view_output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var view_input: texture_storage_2d<rgba16float, read>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
     if any(global_id.xy >= vec2u(view.viewport.zw)) {
         return;
     }

    var color_weight = COLOR_WEIGHT;
    let kernel = array(3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0);
    let current_color = textureLoad(view_input, global_id.xy).rgb;
    let current_albedo = textureLoad(albedo_texture, global_id.xy).rgb;
    let current_normal = textureLoad(normal_texture, global_id.xy).rgb;
    let current_world_position = textureLoad(world_position_texture, global_id.xy).rgb;

    var sum = vec3(0.0);
    var cum_w = 0.0;

    for (var d_x = -2; d_x <= 2; d_x += 1) {
        for (var d_y = -2; d_y <= 2; d_y += 1) {
            let uv = vec2u(clamp(
                vec2i(global_id.xy) + vec2i(d_x, d_y) * i32(step_width),
                vec2i(0),
                vec2i(view.viewport.zw) - vec2i(1)
            ));

            let color = textureLoad(view_input, uv).rgb;
            let d_c = current_color - color;
            let dist_color = dot(d_c, d_c);
            let c_w = min(exp(-(dist_color) / color_weight), 1.0);

            let albedo = textureLoad(albedo_texture, uv).rgb;
            let d_a = current_albedo - albedo;
            let dist_albedo = dot(d_a, d_a);
            let a_w = min(exp(-(dist_albedo) / ALBEDO_WEIGHT), 1.0);

            let normal = textureLoad(normal_texture, uv).rgb;
            let d_n = current_normal - normal;
            let dist_normal = max(dot(d_n, d_n) / f32(step_width), 0.0);
            let n_w = min(exp(-(dist_normal) / NORMAL_WEIGHT), 1.0);

            let world_position = textureLoad(world_position_texture, uv).rgb;
            let d_w_p = current_world_position - world_position;
            let dist_world_position = dot(d_w_p, d_w_p);
            let w_p_w = min(exp(-(dist_world_position) / WORLD_POSITION_WEIGHT), 1.0);

            let weight = c_w * a_w * n_w * w_p_w;

            let kernel_index = max(abs(d_x), abs(d_y));
            sum += color * weight * kernel[kernel_index];
            cum_w += weight * kernel[kernel_index];
        }
    }


    textureStore(view_output, global_id.xy, vec4(sum / max(cum_w, 0.0001), 1.0));
}
