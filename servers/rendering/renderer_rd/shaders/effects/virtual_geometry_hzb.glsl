#[compute]

#version 450

#VERSION_DEFINES

// Builds the hierarchical depth buffer used for occlusion culling of virtual geometry.
// Each texel stores the farthest depth of the footprint it covers. With reversed Z the
// farthest depth is the smallest value, so the reduction is a min().
//
// Every level is stored in one storage buffer, one after the other (see hzb_get_level()
// in virtual_geometry.glsl). A buffer avoids per-mip image views, which not all drivers
// keep apart reliably when the same image is written and sampled in one frame.
//
// Level sizes are halved and rounded down, like regular mipmaps. A texel of level N covers
// 2^(N+1) x 2^(N+1) depth pixels, except the last row and column, which also cover the
// pixels that don't fit (odd sizes), so every pixel is accounted for.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#ifdef MODE_DEPTH
layout(set = 0, binding = 0) uniform sampler2D source_depth;
#endif

layout(set = 0, binding = 1, std430) restrict buffer Hzb {
	float data[];
}
hzb;

layout(push_constant, std430) uniform Params {
	ivec2 source_size;
	ivec2 dest_size;
	uint source_offset; // In floats, unused for MODE_DEPTH.
	uint dest_offset;
	uint pad0;
	uint pad1;
}
params;

float load_source(ivec2 p_pos) {
#ifdef MODE_DEPTH
	return texelFetch(source_depth, p_pos, 0).r;
#else
	return hzb.data[params.source_offset + uint(p_pos.y * params.source_size.x + p_pos.x)];
#endif
}

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pos, params.dest_size))) {
		return;
	}
	ivec2 first = pos * 2;
	// The last texel of each row and column absorbs the leftover source texels.
	ivec2 last = mix(first + 1, params.source_size - 1, equal(pos, params.dest_size - 1));
	last = min(last, params.source_size - 1);
	float depth = 1.0;
	for (int y = first.y; y <= last.y; y++) {
		for (int x = first.x; x <= last.x; x++) {
			depth = min(depth, load_source(ivec2(x, y)));
		}
	}
	hzb.data[params.dest_offset + uint(pos.y * params.dest_size.x + pos.x)] = depth;
}
