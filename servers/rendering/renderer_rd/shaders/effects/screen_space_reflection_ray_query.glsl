#[compute]

#version 460

#VERSION_DEFINES

#extension GL_EXT_ray_query : require

// Ray traced variant of the SSR trace pass (screen_space_reflection.glsl). Reflection rays are
// traced against an acceleration structure of the scene, built in view space, instead of being
// marched through the depth buffer. This finds the exact first hit, including behind other
// objects and for thin geometry, and never reports a hit where there is none.
//
// Hits visible from the camera reuse the lit color of the previous frame, like SSR. Hits that
// are off screen or hidden are shaded with the reflection probes that contain them: a probe
// captured the radiance leaving that very surface, in the direction from its capture origin to
// the hit point, so it acts as a coarse surface cache. Misses, and hits outside every probe,
// write zero validity, so the reflection falls back to the probes and the sky at the shaded
// point, as it does for SSR. The outputs (color with validity in alpha, and mip level) are the
// same as SSR, so filtering, resolve and composition are shared.

#include "../light_data_inc.glsl"
#include "../oct_inc.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_last_frame;
layout(set = 0, binding = 1) uniform sampler2D source_depth;
layout(set = 0, binding = 2) uniform sampler2D source_normal_roughness;
layout(rgba16f, set = 0, binding = 3) uniform restrict writeonly image2D output_color;
layout(r8, set = 0, binding = 4) uniform restrict writeonly image2D output_mip_level;

layout(set = 0, binding = 5, std140) uniform SceneData {
	mat4 projection[2];
	mat4 inv_projection[2];
	mat4 reprojection[2];
	vec4 eye_offset[2];
}
scene_data;

// Instances are placed relative to the camera, so rays are traced in view space.
layout(set = 0, binding = 6) uniform accelerationStructureEXT scene_tlas;

// Reflection probes of this frame, sorted like for the scene shader; local_matrix goes from view space to probe space.
layout(set = 0, binding = 7, std430) restrict readonly buffer ReflectionProbeData {
	ReflectionData data[];
}
reflections;

layout(set = 0, binding = 8) uniform sampler2DArray reflection_atlas;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	int mipmaps;
	float max_distance;
	float pad0;
	float pad1;
	float depth_tolerance;
	bool orthogonal;
	int view_index;
	uint reflection_count;
	vec2 reflection_atlas_border_size;
}
params;

#define M_PI 3.14159265359

vec3 compute_view_pos(vec3 screen_pos) {
	vec4 pos;
	pos.xy = screen_pos.xy * 2.0 - 1.0;
	pos.z = screen_pos.z;
	pos.w = 1.0;
	pos = scene_data.inv_projection[params.view_index] * pos;
	return pos.xyz / pos.w;
}

vec3 compute_screen_pos(vec3 pos) {
	vec4 screen_pos = scene_data.projection[params.view_index] * vec4(pos, 1.0);
	screen_pos.xyz /= screen_pos.w;
	screen_pos.xy = screen_pos.xy * 0.5 + 0.5;
	return screen_pos.xyz;
}

// Same reconstruction as SSR: https://habr.com/ru/articles/744336/
vec3 compute_geometric_normal(ivec2 pixel_pos, float depth_c, vec3 view_c, float pixel_offset) {
	vec4 H = vec4(
			texelFetch(source_depth, pixel_pos + ivec2(-1, 0), 0).x,
			texelFetch(source_depth, pixel_pos + ivec2(-2, 0), 0).x,
			texelFetch(source_depth, pixel_pos + ivec2(1, 0), 0).x,
			texelFetch(source_depth, pixel_pos + ivec2(2, 0), 0).x);

	vec4 V = vec4(
			texelFetch(source_depth, pixel_pos + ivec2(0, -1), 0).x,
			texelFetch(source_depth, pixel_pos + ivec2(0, -2), 0).x,
			texelFetch(source_depth, pixel_pos + ivec2(0, 1), 0).x,
			texelFetch(source_depth, pixel_pos + ivec2(0, 2), 0).x);

	vec2 he = abs((2.0 * H.xz - H.yw) - depth_c);
	vec2 ve = abs((2.0 * V.xz - V.yw) - depth_c);

	int h_sign = he.x < he.y ? -1 : 1;
	int v_sign = ve.x < ve.y ? -1 : 1;

	vec3 view_h = compute_view_pos(vec3((pixel_pos + vec2(h_sign, 0) + pixel_offset) / params.screen_size, H[1 + int(h_sign)]));
	vec3 view_v = compute_view_pos(vec3((pixel_pos + vec2(0, v_sign) + pixel_offset) / params.screen_size, V[1 + int(v_sign)]));

	vec3 h_der = h_sign * (view_h - view_c);
	vec3 v_der = v_sign * (view_v - view_c);

	return cross(v_der, h_der);
}

// Radiance leaving the surface at p_pos (view space) towards the camera, as captured by the probes
// that contain it. Alpha is the coverage of the probes.
vec4 sample_probes_at(vec3 p_pos) {
	vec4 accum = vec4(0.0);
	for (uint i = 0u; i < params.reflection_count && accum.a < 1.0; i++) {
		vec3 box_extents = reflections.data[i].box_extents;
		vec3 local_pos = (reflections.data[i].local_matrix * vec4(p_pos, 1.0)).xyz;
		if (any(greaterThan(abs(local_pos), box_extents)) || reflections.data[i].intensity <= 0.0) {
			continue;
		}

		float blend = 1.0;
		if (reflections.data[i].blend_distance != 0.0) {
			vec3 axis_blend_distance = min(vec3(reflections.data[i].blend_distance), box_extents);
			vec3 blend_axes = (abs(local_pos) - box_extents + axis_blend_distance) / axis_blend_distance;
			blend_axes = clamp(1.0 - blend_axes, vec3(0.0), vec3(1.0));
			blend = pow(blend_axes.x * blend_axes.y * blend_axes.z, 2.0);
		}

		// The probe saw this surface from its capture origin.
		vec3 direction = local_pos - reflections.data[i].box_offset;
		if (dot(direction, direction) < 1e-8) {
			continue;
		}
		vec2 uv = vec3_to_oct_with_border(normalize(direction), params.reflection_atlas_border_size);
		vec3 radiance = textureLod(reflection_atlas, vec3(uv, reflections.data[i].index), 0.0).rgb * reflections.data[i].exposure_normalization;

		float weight = max(0.0, blend - accum.a) * reflections.data[i].intensity;
		accum += vec4(radiance * weight, weight);
	}
	return accum;
}

void main() {
	ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(pixel_pos, params.screen_size))) {
		return;
	}

	vec4 color = vec4(0.0);
	float mip_level = 0.0;

	vec3 screen_pos;
	screen_pos.xy = vec2(pixel_pos + 0.5) / params.screen_size;
	screen_pos.z = texelFetch(source_depth, pixel_pos, 0).x;

	if (screen_pos.z != 0.0) {
		vec3 pos = compute_view_pos(screen_pos);

		vec4 normal_roughness = texelFetch(source_normal_roughness, pixel_pos, 0);
		vec3 normal = normalize(normal_roughness.xyz * 2.0 - 1.0);
		float roughness = normal_roughness.w;
		if (roughness > 0.5) {
			roughness = 1.0 - roughness;
		}
		roughness /= (127.0 / 255.0);

		// Rough reflections come from the probes, like with SSR.
		if (roughness >= 0.7) {
			imageStore(output_color, pixel_pos, vec4(0.0));
			imageStore(output_mip_level, pixel_pos, vec4(0.0));
			return;
		}

		vec3 geom_normal = normalize(compute_geometric_normal(pixel_pos, screen_pos.z, pos, 0.5));

		vec3 view_dir = params.orthogonal ? vec3(0.0, 0.0, -1.0) : normalize(pos - scene_data.eye_offset[params.view_index].xyz);
		vec3 ray_dir = normalize(reflect(view_dir, normal));

		// Normal maps can make the reflection point into the surface: bounce it back out.
		if (dot(ray_dir, geom_normal) < 0.0) {
			ray_dir = normalize(reflect(ray_dir, geom_normal));
		}

		// Start slightly above the surface, scaled with distance to follow depth precision.
		float bias = max(0.002, abs(pos.z) * 0.002);
		vec3 origin = pos + geom_normal * bias;

		rayQueryEXT ray_query;
		rayQueryInitializeEXT(ray_query, scene_tlas, gl_RayFlagsOpaqueEXT, 0xFF, origin, 0.0, ray_dir, params.max_distance);
		while (rayQueryProceedEXT(ray_query)) {
		}

		float ray_len = 1.0; // In screen units, like SSR. Misses and hidden hits use a long ray for the blur.
		if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
			vec3 hit_pos = origin + ray_dir * rayQueryGetIntersectionTEXT(ray_query, true);
			vec3 hit_screen_pos = compute_screen_pos(hit_pos);

			float screen_validity = 0.0;
			bool in_front = params.orthogonal || hit_pos.z < 0.0;
			if (in_front && all(greaterThanEqual(hit_screen_pos.xy, vec2(0.0))) && all(lessThan(hit_screen_pos.xy, vec2(1.0)))) {
				ray_len = length(hit_screen_pos.xy - screen_pos.xy);

				// The hit is only usable if the camera sees it: nothing may be in front of it.
				ivec2 hit_pixel = ivec2(hit_screen_pos.xy * params.screen_size);
				float scene_depth = texelFetch(source_depth, hit_pixel, 0).x;
				vec3 scene_pos = compute_view_pos(vec3(hit_screen_pos.xy, scene_depth));
				float tolerance = max(params.depth_tolerance, abs(hit_pos.z) * 0.01);
				float occlusion = scene_pos.z - hit_pos.z; // Positive when the scene is in front of the hit.
				float validity = scene_depth == 0.0 ? 0.0 : 1.0 - smoothstep(0.0, tolerance, occlusion);

				vec4 reprojected_pos;
				reprojected_pos.xy = hit_screen_pos.xy * 2.0 - 1.0;
				reprojected_pos.z = hit_screen_pos.z;
				reprojected_pos.w = 1.0;
				reprojected_pos = scene_data.reprojection[params.view_index] * reprojected_pos;
				reprojected_pos.xy = reprojected_pos.xy / reprojected_pos.w * 0.5 + 0.5;

				// Fade towards the screen edges, where the previous frame has no data.
				vec2 reprojected_pixel_pos = reprojected_pos.xy * params.screen_size;
				vec2 margin = vec2((params.screen_size.x + params.screen_size.y) * 0.05);
				vec2 margin_grad = mix(params.screen_size - reprojected_pixel_pos, reprojected_pixel_pos, lessThan(reprojected_pixel_pos, params.screen_size * 0.5));
				float margin_blend = smoothstep(0.0, margin.x * margin.y, margin_grad.x * margin_grad.y);

				// Unlike SSR, no fade in or out with the ray length: those hide the artifacts of marching
				// the depth buffer, and a ray traced hit is exact.
				screen_validity = validity * margin_blend;
				if (screen_validity > 0.0) {
					color = vec4(textureLod(source_last_frame, reprojected_pos.xy, 0).xyz, 1.0) * screen_validity;
				}
			}

			if (screen_validity < 1.0) {
				// Off screen, hidden, or faded out on screen: complete with the probes around the hit.
				vec4 probes = sample_probes_at(hit_pos);
				float probe_validity = min(probes.a, 1.0) * (1.0 - screen_validity);
				if (probe_validity > 0.0) {
					color += vec4(probes.rgb / probes.a, 1.0) * probe_validity;
				}
			}

			// Tone map like SSR, for smoother roughness filtering across samples of varying luminance.
			const vec3 rec709_luminance_weights = vec3(0.2126, 0.7152, 0.0722);
			color.rgb /= 1.0 + dot(color.rgb, rec709_luminance_weights);
		}

		if (roughness > 0.001) {
			// Same cone approximation as SSR.
			float cone_angle = min(roughness, 0.999) * M_PI * 0.5;
			float cone_len = max(ray_len, 1e-4);
			float op_len = 2.0 * tan(cone_angle) * cone_len;
			float a = op_len;
			float h = cone_len;
			float blur_radius = (a * (sqrt(a * a + 4.0 * h * h) - a)) / (4.0 * h);
			mip_level = clamp(log2(blur_radius * max(params.screen_size.x, params.screen_size.y) / 16.0 + 1.0), 0, params.mipmaps - 1);
		}
		mip_level *= pow(clamp(1.25 - ray_len, 0.0, 1.0), 0.2);
	}

	imageStore(output_color, pixel_pos, color);
	imageStore(output_mip_level, pixel_pos, vec4(mip_level / 14.0, 0.0, 0.0, 0.0));
}
