#import bevy_render::view::View

// https://www.shadertoy.com/view/4dfGDH

const SIGMA: f32 = 10.0;
const BSIGMA: f32 = 0.1;
const MSIZE: u32 = 15;

@group(0) @binding(0) var view_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var view_input: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var<uniform> view: View;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.viewport.zw)) {
        return;
    }

    let original_color = textureLoad(view_input, global_id.xy).rgb;

    let k_size = (MSIZE - 1) / 2;
    var kernel = array<f32, MSIZE>();
    var final_color = vec3(0.0);
    for (var j = u32(0); j <= k_size; j++) {
        let norm = normpdf(f32(j), SIGMA);
        kernel[k_size + j] = norm;
        kernel[k_size - j] = norm;
    }

    var Z = 0.0;

    var color = vec3(0.0);
    var factor = 0.0;
    let bZ = 1.0 / normpdf(0.0, BSIGMA);
    for (var i = -i32(k_size); i <= i32(k_size); i++) {
        for (var j = -i32(k_size); j <= i32(k_size); j++) {
            color = textureLoad(view_input, vec2<u32>(vec2<f32>(global_id.xy) + vec2(f32(i), f32(j)))).rgb;
            factor = normpdf3(color - original_color, BSIGMA) * bZ * kernel[u32(i32(k_size) + j)] * kernel[u32(i32(k_size) + i)];
            Z += factor;
            final_color += color * factor;
        }
    }

    textureStore(view_output, global_id.xy, vec4(final_color / Z, 1.0));
}

fn normpdf(x: f32, sigma: f32) -> f32 {
    return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

fn normpdf3(v: vec3<f32>, sigma: f32) -> f32 {
    return 0.39894 * exp(-0.5 * dot(v, v) / (sigma * sigma)) / sigma;
}