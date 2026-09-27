#[compute]

#version 460

#VERSION_DEFINES

#extension GL_EXT_ray_query : require

// Ray traced ambient occlusion, used instead of SSAO when meshes have acceleration structures.
// Short rays are traced from every pixel into the hemisphere around its normal, against an
// acceleration structure of the scene built in view space. Unlike SSAO, occluders outside the
// screen or hidden behind other objects are taken into account, and there are no halos around
// objects in front of others. The result is written to the same texture as SSAO, so the scene
// shader applies it the same way.
//
// MODE_TRACE:    a couple of rays per pixel, with a pattern that changes every frame.
// MODE_TEMPORAL: accumulates the result over frames: the history is reprojected with the
//                camera motion and rejected where the depth it was computed for doesn't match
//                (disocclusions).
// MODE_BLUR:     separable bilateral blur that preserves depth and normal discontinuities.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_depth;
layout(set = 0, binding = 1) uniform sampler2D source_normal_roughness;

#ifdef MODE_TRACE
layout(set = 0, binding = 2) uniform accelerationStructureEXT scene_tlas;
#else
layout(set = 0, binding = 2) uniform sampler2D source_ao; // Occlusion in the red channel.
#endif

#ifdef MODE_TEMPORAL
layout(set = 0, binding = 3) uniform sampler2D source_history; // Occlusion, linear depth.
layout(rg16f, set = 0, binding = 4) uniform restrict writeonly image2D dest_history;
layout(set = 0, binding = 5, std140) uniform Reprojection {
	mat4 current_to_previous_view;
	mat4 previous_projection;
}
reprojection;
#else
layout(r8, set = 0, binding = 3) uniform restrict writeonly image2D dest_ao;
#endif

#define FLAG_BLUR_VERTICAL 1u
#define FLAG_HISTORY_VALID 2u

layout(push_constant, std430) uniform Params {
	mat4 inv_projection;
	ivec2 screen_size;
	float radius;
	float intensity;
	float power;
	uint frame;
	uint ray_count;
	uint flags;
}
params;

vec3 compute_view_pos(ivec2 p_pixel, float p_depth) {
	vec4 pos = vec4((vec2(p_pixel) + 0.5) / vec2(params.screen_size) * 2.0 - 1.0, p_depth, 1.0);
	pos = params.inv_projection * pos;
	return pos.xyz / pos.w;
}

vec3 load_normal(ivec2 p_pixel) {
	return normalize(texelFetch(source_normal_roughness, p_pixel, 0).xyz * 2.0 - 1.0);
}

#ifdef MODE_TRACE

uint hash(uint p_value) {
	// PCG hash.
	uint state = p_value * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

float random(inout uint r_seed) {
	r_seed = hash(r_seed);
	return float(r_seed) / 4294967295.0;
}

// Orthonormal basis without branches (Duff et al. 2017).
void make_basis(vec3 p_normal, out vec3 r_tangent, out vec3 r_bitangent) {
	float s = p_normal.z >= 0.0 ? 1.0 : -1.0;
	float a = -1.0 / (s + p_normal.z);
	float b = p_normal.x * p_normal.y * a;
	r_tangent = vec3(1.0 + s * p_normal.x * p_normal.x * a, s * b, -s * p_normal.x);
	r_bitangent = vec3(b, s + p_normal.y * p_normal.y * a, -p_normal.y);
}

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.screen_size))) {
		return;
	}

	float depth = texelFetch(source_depth, pixel, 0).r;
	if (depth == 0.0) {
		imageStore(dest_ao, pixel, vec4(1.0)); // Sky.
		return;
	}

	vec3 pos = compute_view_pos(pixel, depth);
	vec3 normal = load_normal(pixel);
	vec3 tangent, bitangent;
	make_basis(normal, tangent, bitangent);

	// Start above the surface, scaled with distance to follow depth precision.
	vec3 origin = pos + normal * max(0.002, abs(pos.z) * 0.002);

	uint seed = hash(uint(pixel.x) + uint(pixel.y) * uint(params.screen_size.x)) ^ hash(params.frame);
	float occlusion = 0.0;
	for (uint i = 0u; i < params.ray_count; i++) {
		// Cosine weighted direction: the estimate converges to the cosine weighted visibility.
		float u1 = random(seed);
		float u2 = random(seed);
		float r = sqrt(u1);
		float phi = 6.28318530718 * u2;
		vec3 direction = tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + normal * sqrt(max(0.0, 1.0 - u1));

		rayQueryEXT ray_query;
		rayQueryInitializeEXT(ray_query, scene_tlas, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.0, direction, params.radius);
		while (rayQueryProceedEXT(ray_query)) {
		}
		if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
			// Fade out towards the radius, so the occlusion doesn't end with a hard edge.
			float t = rayQueryGetIntersectionTEXT(ray_query, true) / params.radius;
			occlusion += 1.0 - smoothstep(0.5, 1.0, t);
		}
	}

	float visibility = 1.0 - occlusion / float(params.ray_count);
	float ao = pow(clamp(visibility, 0.0, 1.0), max(params.intensity * params.power * 0.5, 0.01));
	imageStore(dest_ao, pixel, vec4(ao));
}

#endif // MODE_TRACE

#ifdef MODE_TEMPORAL

// Weight of the new frame: the result converges over about 1 / weight frames.
#define TEMPORAL_WEIGHT 0.12

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.screen_size))) {
		return;
	}

	float depth = texelFetch(source_depth, pixel, 0).r;
	if (depth == 0.0) {
		imageStore(dest_history, pixel, vec4(1.0, 0.0, 0.0, 0.0));
		return;
	}

	vec3 pos = compute_view_pos(pixel, depth);
	float current = texelFetch(source_ao, pixel, 0).r;
	float result = current;

	if ((params.flags & FLAG_HISTORY_VALID) != 0u) {
		vec4 previous_view = reprojection.current_to_previous_view * vec4(pos, 1.0);
		vec4 previous_clip = reprojection.previous_projection * previous_view;
		if (previous_clip.w > 0.0) {
			vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
			if (all(greaterThanEqual(previous_uv, vec2(0.0))) && all(lessThanEqual(previous_uv, vec2(1.0)))) {
				vec2 history = textureLod(source_history, previous_uv, 0.0).rg;
				float expected_depth = -previous_view.z;
				// Different depth: that point wasn't visible in the previous frame.
				if (abs(history.g - expected_depth) < expected_depth * 0.03) {
					result = mix(history.r, current, TEMPORAL_WEIGHT);
				}
			}
		}
	}

	imageStore(dest_history, pixel, vec4(result, -pos.z, 0.0, 0.0));
}

#endif // MODE_TEMPORAL

#ifdef MODE_BLUR

#define BLUR_RADIUS 4

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.screen_size))) {
		return;
	}

	float depth = texelFetch(source_depth, pixel, 0).r;
	if (depth == 0.0) {
		imageStore(dest_ao, pixel, vec4(1.0));
		return;
	}

	vec3 pos = compute_view_pos(pixel, depth);
	vec3 normal = load_normal(pixel);
	ivec2 step_dir = (params.flags & FLAG_BLUR_VERTICAL) != 0u ? ivec2(0, 1) : ivec2(1, 0);
	float depth_scale = 1.0 / max(abs(pos.z) * 0.02, 0.001);

	float sum = 0.0;
	float weight_sum = 0.0;
	for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; i++) {
		ivec2 sample_pixel = clamp(pixel + step_dir * i, ivec2(0), params.screen_size - 1);
		float sample_depth = texelFetch(source_depth, sample_pixel, 0).r;
		if (sample_depth == 0.0) {
			continue;
		}
		vec3 sample_pos = compute_view_pos(sample_pixel, sample_depth);
		float w = exp(-float(i * i) / (2.0 * 2.5 * 2.5));
		w *= exp(-abs(sample_pos.z - pos.z) * depth_scale);
		w *= pow(max(dot(normal, load_normal(sample_pixel)), 0.0), 8.0);
		sum += texelFetch(source_ao, sample_pixel, 0).r * w;
		weight_sum += w;
	}

	imageStore(dest_ao, pixel, vec4(weight_sum > 0.0 ? sum / weight_sum : 1.0));
}

#endif // MODE_BLUR
