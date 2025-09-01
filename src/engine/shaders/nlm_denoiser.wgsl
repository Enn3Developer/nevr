// https://www.shadertoy.com/view/tt2SWK

#import bevy_render::view::View

const DIST: f32 = 0.5;

@group(0) @binding(0) var view_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var view_input: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(2) var<uniform> view: View;

@compute @workgroup_size(8, 8, 1)
fn vertical(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.viewport.zw)) {
        return;
    }

    let original_color = textureLoad(view_input, global_id.xy);
    var acc = original_color;
    var count = 1.0;
    var ra = original_color.a;
    let size = i32(view.viewport.w) / 8;

    for (var y_off = -1; y_off > (-size - 1); y_off--) {
        let weight = 1.0 - abs(f32(y_off)) / f32(size);
        let y_crd = vec2<i32>(global_id.xy) + vec2(0, y_off);
        let color = textureLoad(view_input, y_crd);
        let dist = abs(color.a - ra);

        if (dist < DIST) {
            acc += vec4(color.rgb * weight, acc.a);
            count += weight;
            ra = color.a;
        }
    }

    ra = original_color.a;

    for (var y_off = 1; y_off < (size + 1); y_off++) {
        let weight = 1.0 - abs(f32(y_off)) / f32(size);
        let y_crd = vec2<i32>(global_id.xy) + vec2(0, y_off);
        let color = textureLoad(view_input, y_crd);
        let dist = abs(color.a - ra);

        if (dist < DIST) {
            acc += vec4(color.rgb * weight, acc.a);
            count += weight;
            ra = color.a;
        }
    }

    if (count <= 1.0) {
        acc = vec4(acc.rgb + 0.25 * textureLoad(view_input, vec2<i32>(global_id.xy) + vec2(0, -2)).rgb, acc.a);
        acc = vec4(acc.rgb + 0.5 * textureLoad(view_input, vec2<i32>(global_id.xy) + vec2(0, -1)).rgb, acc.a);
        acc = vec4(acc.rgb + 0.5 * textureLoad(view_input, vec2<i32>(global_id.xy) + vec2(0, 1)).rgb, acc.a);
        acc = vec4(acc.rgb + 0.25 * textureLoad(view_input, vec2<i32>(global_id.xy) + vec2(0, 2)).rgb, acc.a);

        count += 1.5;
    }

    textureStore(view_input, global_id.xy, vec4((acc / count).rgb, original_color.a));
}

@compute @workgroup_size(8, 8, 1)
fn horizontal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.viewport.zw)) {
        return;
    }

    let original_color = textureLoad(view_input, global_id.xy);
    var acc = original_color;
    var count = 1.0;
    var ra = original_color.a;
    let size = i32(view.viewport.z) / 8;

    for (var x_off = -1; x_off > (-size - 1); x_off--) {
        let weight = 1.0 - abs(f32(x_off)) / f32(size);
        let x_crd = vec2<i32>(global_id.xy) + vec2(x_off, 0);
        let color = textureLoad(view_input, x_crd);
        let dist = abs(color.a - ra);

        if (dist < DIST) {
            acc += vec4(color.rgb * weight, acc.a);
            count += weight;
            ra = color.a;
        }
    }

    ra = original_color.a;

    for (var x_off = 1; x_off < (size + 1); x_off++) {
        let weight = 1.0 - abs(f32(x_off)) / f32(size);
        let x_crd = vec2<i32>(global_id.xy) + vec2(x_off, 0);
        let color = textureLoad(view_input, x_crd);
        let dist = abs(color.a - ra);

        if (dist < DIST) {
            acc += vec4(color.rgb * weight, acc.a);
            count += weight;
            ra = color.a;
        }
    }

    if (count <= 1.0) {
        acc = vec4(acc.rgb + 0.25 * textureLoad(view_input, vec2<i32>(global_id.xy) + vec2(-2, 0)).rgb, acc.a);
        acc = vec4(acc.rgb + 0.5 * textureLoad(view_input, vec2<i32>(global_id.xy) + vec2(-1, 0)).rgb, acc.a);
        acc = vec4(acc.rgb + 0.5 * textureLoad(view_input, vec2<i32>(global_id.xy) + vec2(1, 0)).rgb, acc.a);
        acc = vec4(acc.rgb + 0.25 * textureLoad(view_input, vec2<i32>(global_id.xy) + vec2(2, 0)).rgb, acc.a);

        count += 1.5;
    }

    textureStore(view_output, global_id.xy, vec4((acc / count).rgb, original_color.a));
}