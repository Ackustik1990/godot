#[compute]

#version 450

#VERSION_DEFINES

// GPU-driven cluster LOD selection and culling for virtual geometry.
//
// MODE_CULL:   one thread per cluster of every job (instance/view pair). Selects the
//              clusters of the LOD cut, culls them against the view frustum and their
//              normal cone, appends the survivors to a visible list and accumulates the
//              number of indices needed per job. With occlusion culling, clusters hidden
//              behind the depth of the previous frame (reprojected) are deferred to the
//              occluded list instead.
// MODE_CULL_OCCLUDED: after the depth pre-pass, tests the occluded list against the depth
//              of the current frame. Clusters that turn out to be visible (disocclusions,
//              moving objects) are drawn in a second pass. Together, both passes never
//              cull a visible cluster.
// MODE_PREFIX: a single workgroup computes where the indices of each job go in the
//              shared output index buffer (exclusive prefix sum) and prepares the
//              indirect dispatch of MODE_EMIT. The second pass appends its indices
//              after the ones of the first pass.
// MODE_EMIT:   one workgroup per visible cluster, one thread per triangle. Decodes the
//              cluster's micro index buffer and writes absolute vertex indices into the
//              output index buffer, which is then drawn with one indirect draw per job.

#if defined(MODE_EMIT)
#define WORKGROUP_SIZE 128
#elif defined(MODE_PREFIX)
#define WORKGROUP_SIZE 256
#else
#define WORKGROUP_SIZE 64 // MODE_CULL and MODE_CULL_OCCLUDED.
#endif

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

#define ROOT_ERROR 1.0e38
#define COMMAND_STRIDE 8u
#define DISPATCH_WIDTH 32768u

// Must match VirtualGeometryFormat::Cluster.
struct Cluster {
	vec4 bounds;
	vec4 lod_sphere;
	vec4 parent_sphere;
	float lod_error;
	float parent_error;
	uint cone;
	uint counts;
	uint vertex_offset;
	uint triangle_offset;
	uint pad0;
	uint pad1;
};

// Must match VirtualGeometry::JobGPU.
struct Job {
	vec4 transform_x; // Rows of the 3x4 mesh-to-view-origin transform.
	vec4 transform_y;
	vec4 transform_z;
	uint cluster_offset;
	uint cluster_count;
	uint data_offset;
	uint workgroup_offset;
	uint view_index;
	uint flags;
	float scale;
	uint pad;
};

#define JOB_FLAG_CONE_CULL 1u
#define JOB_FLAG_CONE_INVERT 2u

// Must match VirtualGeometry::ViewGPU. Positions are relative to the view origin.
struct View {
	vec4 planes[6]; // Outward facing: a sphere is outside when dot(n, c) - d > r.
	vec4 lod_position; // xyz: LOD camera position, w: LOD factor (1 / (distance multiplier * threshold)).
	vec4 cull_position; // xyz: camera position (perspective) or view direction (orthogonal).
	mat4 occlusion_previous; // To the clip space of the previous frame (reversed Z, depth in [0, 1]).
	mat4 occlusion_current; // To the clip space of the current frame.
	vec4 hzb_size; // xy: depth buffer size in pixels, z: HZB mip count.
	uint flags;
	uint plane_mask;
	uint pad0;
	uint pad1;
};

#define VIEW_FLAG_LOD_ORTHOGONAL 1u
#define VIEW_FLAG_CULL_ORTHOGONAL 2u
#define VIEW_FLAG_FRUSTUM 4u
#define VIEW_FLAG_CONE 8u
#define VIEW_FLAG_FORCE_LOD0 16u
#define VIEW_FLAG_OCCLUSION_PREVIOUS 32u

layout(set = 0, binding = 0, std430) restrict readonly buffer Clusters {
	Cluster data[];
}
clusters;

layout(set = 0, binding = 1, std430) restrict readonly buffer ClusterData {
	uint data[];
}
cluster_data;

layout(set = 0, binding = 2, std430) restrict readonly buffer Jobs {
	Job data[];
}
jobs;

layout(set = 0, binding = 3, std430) restrict readonly buffer Views {
	View data[];
}
views;

// Per job: index_count, instance_count, first_index, vertex_offset, first_instance (VkDrawIndexedIndirectCommand), write cursor, pad, pad.
layout(set = 0, binding = 4, std430) restrict buffer DrawCommands {
	uint data[];
}
draw_commands;

layout(set = 0, binding = 5, std430) restrict buffer VisibleClusters {
	uvec2 data[]; // x: job, y: cluster (relative to the job).
}
visible_clusters;

layout(set = 0, binding = 6, std430) restrict buffer BatchState {
	uint visible_count;
	uint index_count; // End of the indices required by the batch up to this pass, even if they didn't fit.
	uint dispatch_x;
	uint dispatch_y;
	uint dispatch_z;
	uint overflow;
	uint index_base; // Where the indices of this pass start in the output buffer.
	uint pad;
}
batch;

layout(set = 0, binding = 7, std430) restrict writeonly buffer OutputIndices {
	uint data[];
}
output_indices;

// Clusters rejected by the occlusion test of the first pass, retested in the second pass.
layout(set = 0, binding = 8, std430) restrict buffer OccludedClusters {
	uint count;
	uint dispatch_x;
	uint dispatch_y;
	uint dispatch_z;
	uint index_end; // End of the indices written by the first pass.
	uint pad0;
	uint pad1;
	uint pad2;
	uvec2 data[];
}
occluded;

// Every level of the HZB, one after the other (see virtual_geometry_hzb.glsl).
layout(set = 0, binding = 9, std430) restrict readonly buffer Hzb {
	float data[];
}
hzb;

#define PARAMS_FLAG_APPEND 1u

layout(push_constant, std430) uniform Params {
	uint job_count;
	uint workgroup_count;
	uint index_capacity;
	uint visible_capacity;
	uint flags;
	uint pad0;
	uint pad1;
	uint pad2;
}
params;

#if defined(MODE_CULL) || defined(MODE_CULL_OCCLUDED)

// Returns true when the sphere is hidden behind the depth stored in the HZB.
// The sphere is projected through its bounding box, which is conservative.
// With p_outside_visible, a box that is partly outside the view is never occluded: the
// HZB of the previous frame knows nothing about what is there now.
bool is_occluded(mat4 p_matrix, vec3 p_center, float p_radius, vec4 p_hzb_size, bool p_outside_visible) {
	vec2 uv_min = vec2(1.0);
	vec2 uv_max = vec2(0.0);
	float closest = 0.0;
	for (uint i = 0u; i < 8u; i++) {
		vec3 corner = p_center + vec3((i & 1u) != 0u ? p_radius : -p_radius, (i & 2u) != 0u ? p_radius : -p_radius, (i & 4u) != 0u ? p_radius : -p_radius);
		vec4 clip = p_matrix * vec4(corner, 1.0);
		if (clip.w <= 1e-5) {
			return false; // Crosses the camera plane, can't be tested.
		}
		vec3 ndc = clip.xyz / clip.w;
		vec2 uv = ndc.xy * 0.5 + 0.5;
		uv_min = min(uv_min, uv);
		uv_max = max(uv_max, uv);
		closest = max(closest, ndc.z); // Reversed Z: larger is closer.
	}
	if (closest >= 1.0) {
		return false; // Crosses the near plane.
	}
	if (p_outside_visible && (any(lessThan(uv_min, vec2(0.0))) || any(greaterThan(uv_max, vec2(1.0))))) {
		return false;
	}

	// In depth buffer pixels, with a margin for the jitter of temporal antialiasing.
	vec2 pixel_min = clamp(uv_min, 0.0, 1.0) * p_hzb_size.xy - 1.0;
	vec2 pixel_max = clamp(uv_max, 0.0, 1.0) * p_hzb_size.xy + 1.0;
	vec2 extent = pixel_max - pixel_min;

	// Texels of level N cover 2^(N+1) pixels: pick the level where the rectangle spans at most 2x2 texels.
	int level = int(clamp(ceil(log2(max(max(extent.x, extent.y), 1.0))) - 1.0, 0.0, p_hzb_size.z - 1.0));

	// Find where the level starts; sizes are halved and rounded down from half the depth buffer size.
	ivec2 size = max(ivec2(p_hzb_size.xy) >> 1, ivec2(1));
	uint offset = 0u;
	for (int i = 0; i < level; i++) {
		offset += uint(size.x * size.y);
		size = max(size >> 1, ivec2(1));
	}

	// The last row and column of a level also cover the pixels that don't fit, hence the clamp.
	ivec2 t0 = clamp(ivec2(max(pixel_min, vec2(0.0))) >> (level + 1), ivec2(0), size - 1);
	ivec2 t1 = clamp(ivec2(max(pixel_max, vec2(0.0))) >> (level + 1), ivec2(0), size - 1);
	uint row0 = offset + uint(t0.y * size.x);
	uint row1 = offset + uint(t1.y * size.x);
	float farthest = min(min(hzb.data[row0 + uint(t0.x)], hzb.data[row0 + uint(t1.x)]), min(hzb.data[row1 + uint(t0.x)], hzb.data[row1 + uint(t1.x)]));
	return closest < farthest;
}

#endif

#ifdef MODE_CULL

shared uint s_visible_count;
shared uint s_index_count;
shared uint s_visible_base;
shared uint s_occluded_count;
shared uint s_occluded_base;

vec3 transform_point(Job p_job, vec3 p_point) {
	vec4 point = vec4(p_point, 1.0);
	return vec3(dot(p_job.transform_x, point), dot(p_job.transform_y, point), dot(p_job.transform_z, point));
}

vec3 transform_vector(Job p_job, vec3 p_vector) {
	return vec3(dot(p_job.transform_x.xyz, p_vector), dot(p_job.transform_y.xyz, p_vector), dot(p_job.transform_z.xyz, p_vector));
}

// True when the simplification error of a group projects below the pixel threshold.
// Group spheres enclose their children and errors are monotonic along the DAG, so for
// any camera a parent is never "fine enough" while one of its children isn't.
bool is_fine_enough(View p_view, Job p_job, vec4 p_sphere, float p_error) {
	if (p_error <= 0.0) {
		return true; // Full detail.
	}
	if (p_error >= ROOT_ERROR || (p_view.flags & VIEW_FLAG_FORCE_LOD0) != 0u) {
		return false;
	}
	float error = p_error * p_job.scale * p_view.lod_position.w;
	if ((p_view.flags & VIEW_FLAG_LOD_ORTHOGONAL) != 0u) {
		return error <= 1.0;
	}
	vec3 center = transform_point(p_job, p_sphere.xyz);
	float distance = length(center - p_view.lod_position.xyz) - p_sphere.w * p_job.scale;
	// When the camera is inside the sphere, distance is negative and only full detail passes.
	return error <= distance;
}

void main() {
	uint workgroup = gl_WorkGroupID.y * DISPATCH_WIDTH + gl_WorkGroupID.x;
	if (workgroup >= params.workgroup_count) {
		return; // Uniform for the whole workgroup.
	}

	// Find the job owning this workgroup; jobs are sorted by workgroup offset.
	uint lo = 0u;
	uint hi = params.job_count - 1u;
	while (lo < hi) {
		uint mid = (lo + hi + 1u) >> 1u;
		if (jobs.data[mid].workgroup_offset <= workgroup) {
			lo = mid;
		} else {
			hi = mid - 1u;
		}
	}
	uint job_index = lo;
	Job job = jobs.data[job_index];

	if (gl_LocalInvocationIndex == 0u) {
		s_visible_count = 0u;
		s_index_count = 0u;
		s_occluded_count = 0u;
	}
	barrier();

	uint local_cluster = (workgroup - job.workgroup_offset) * WORKGROUP_SIZE + gl_LocalInvocationIndex;
	bool visible = false;
	bool deferred = false;
	uint index_count = 0u;

	if (local_cluster < job.cluster_count) {
		Cluster cluster = clusters.data[job.cluster_offset + local_cluster];
		View view = views.data[job.view_index];

		// LOD cut: this cluster is detailed enough, but the group it simplifies into isn't.
		visible = is_fine_enough(view, job, cluster.lod_sphere, cluster.lod_error) && !is_fine_enough(view, job, cluster.parent_sphere, cluster.parent_error);

		if (visible) {
			vec3 center = transform_point(job, cluster.bounds.xyz);
			float radius = cluster.bounds.w * job.scale;

			if ((view.flags & VIEW_FLAG_FRUSTUM) != 0u) {
				for (uint i = 0u; i < 6u; i++) {
					if ((view.plane_mask & (1u << i)) != 0u && dot(view.planes[i].xyz, center) - view.planes[i].w > radius) {
						visible = false;
						break;
					}
				}
			}

			if (visible && (view.flags & VIEW_FLAG_CONE) != 0u && (job.flags & JOB_FLAG_CONE_CULL) != 0u) {
				vec4 cone = unpackSnorm4x8(cluster.cone);
				if (cone.w < 1.0) {
					// The quantized cutoff is conservative for the quantized (not renormalized) axis,
					// so only undo the uniform scale of the instance.
					vec3 axis = transform_vector(job, cone.xyz) / job.scale;
					if ((job.flags & JOB_FLAG_CONE_INVERT) != 0u) {
						axis = -axis;
					}
					if ((view.flags & VIEW_FLAG_CULL_ORTHOGONAL) != 0u) {
						// cull_position holds the view direction.
						visible = dot(view.cull_position.xyz, axis) < cone.w;
					} else {
						vec3 to_cluster = center - view.cull_position.xyz;
						visible = dot(to_cluster, axis) < cone.w * length(to_cluster) + radius;
					}
				}
			}

			if (visible && (view.flags & VIEW_FLAG_OCCLUSION_PREVIOUS) != 0u && is_occluded(view.occlusion_previous, center, radius, view.hzb_size, true)) {
				// Hidden last frame: retest it against this frame's depth after the pre-pass.
				visible = false;
				deferred = true;
			}
		}

		if (visible) {
			index_count = ((cluster.counts & 0xFFu) + 1u) * 3u;
		}
	}

	// Aggregate in shared memory so each workgroup does only one global atomic per counter.
	uint slot = 0u;
	if (visible) {
		slot = atomicAdd(s_visible_count, 1u);
		atomicAdd(s_index_count, index_count);
	} else if (deferred) {
		slot = atomicAdd(s_occluded_count, 1u);
	}
	barrier();

	if (gl_LocalInvocationIndex == 0u) {
		s_visible_base = 0u;
		if (s_visible_count > 0u) {
			s_visible_base = atomicAdd(batch.visible_count, s_visible_count);
			atomicAdd(draw_commands.data[job_index * COMMAND_STRIDE], s_index_count);
		}
		s_occluded_base = 0u;
		if (s_occluded_count > 0u) {
			s_occluded_base = atomicAdd(occluded.count, s_occluded_count);
		}
	}
	barrier();

	if (visible) {
		uint index = s_visible_base + slot;
		if (index < params.visible_capacity) {
			visible_clusters.data[index] = uvec2(job_index, local_cluster);
		}
	} else if (deferred) {
		uint index = s_occluded_base + slot;
		if (index < params.visible_capacity) {
			occluded.data[index] = uvec2(job_index, local_cluster);
		}
	}
}

#endif // MODE_CULL

#ifdef MODE_CULL_OCCLUDED

shared uint s_visible_count;
shared uint s_visible_base;

void main() {
	uint entry_index = (gl_WorkGroupID.y * DISPATCH_WIDTH + gl_WorkGroupID.x) * WORKGROUP_SIZE + gl_LocalInvocationIndex;

	if (gl_LocalInvocationIndex == 0u) {
		s_visible_count = 0u;
	}
	barrier();

	bool visible = false;
	uvec2 entry = uvec2(0u);
	if (entry_index < min(occluded.count, params.visible_capacity)) {
		entry = occluded.data[entry_index];
		Job job = jobs.data[entry.x];
		Cluster cluster = clusters.data[job.cluster_offset + entry.y];
		View view = views.data[job.view_index];
		vec4 point = vec4(cluster.bounds.xyz, 1.0);
		vec3 center = vec3(dot(job.transform_x, point), dot(job.transform_y, point), dot(job.transform_z, point));
		// Parts outside the view are culled by the frustum, so only the visible part is tested.
		visible = !is_occluded(view.occlusion_current, center, cluster.bounds.w * job.scale, view.hzb_size, false);
		if (visible) {
			// Entries of a workgroup belong to different jobs, so count per thread.
			atomicAdd(draw_commands.data[entry.x * COMMAND_STRIDE], ((cluster.counts & 0xFFu) + 1u) * 3u);
		}
	}

	uint slot = 0u;
	if (visible) {
		slot = atomicAdd(s_visible_count, 1u);
	}
	barrier();

	if (gl_LocalInvocationIndex == 0u) {
		s_visible_base = s_visible_count > 0u ? atomicAdd(batch.visible_count, s_visible_count) : 0u;
	}
	barrier();

	if (visible) {
		uint index = s_visible_base + slot;
		if (index < params.visible_capacity) {
			visible_clusters.data[index] = entry;
		}
	}
}

#endif // MODE_CULL_OCCLUDED

#ifdef MODE_PREFIX

shared uint s_scan[WORKGROUP_SIZE];
shared uint s_running;

void main() {
	uint lid = gl_LocalInvocationIndex;
	bool append = (params.flags & PARAMS_FLAG_APPEND) != 0u;
	if (lid == 0u) {
		s_running = append ? min(occluded.index_end, params.index_capacity) : 0u;
		batch.index_base = s_running;
	}
	barrier();

	// Exclusive prefix sum of the index counts, in chunks of WORKGROUP_SIZE jobs.
	for (uint first = 0u; first < params.job_count; first += WORKGROUP_SIZE) {
		uint job_index = first + lid;
		uint count = job_index < params.job_count ? draw_commands.data[job_index * COMMAND_STRIDE] : 0u;
		s_scan[lid] = count;
		barrier();

		for (uint offset = 1u; offset < WORKGROUP_SIZE; offset <<= 1u) {
			uint value = lid >= offset ? s_scan[lid - offset] : 0u;
			barrier();
			s_scan[lid] += value;
			barrier();
		}

		if (job_index < params.job_count) {
			draw_commands.data[job_index * COMMAND_STRIDE + 2u] = s_running + s_scan[lid] - count;
		}
		barrier();

		if (lid == WORKGROUP_SIZE - 1u) {
			s_running += s_scan[lid];
		}
		barrier();
	}

	if (lid == 0u) {
		batch.index_count = s_running;
		uint visible = min(batch.visible_count, params.visible_capacity);
		batch.dispatch_x = min(visible, DISPATCH_WIDTH);
		batch.dispatch_y = (visible + DISPATCH_WIDTH - 1u) / DISPATCH_WIDTH;
		batch.dispatch_z = 1u;

		if (!append) {
			// Arguments of the second pass (one thread per occluded cluster, 64 per workgroup).
			uint occluded_groups = (min(occluded.count, params.visible_capacity) + 63u) / 64u;
			occluded.dispatch_x = min(occluded_groups, DISPATCH_WIDTH);
			occluded.dispatch_y = (occluded_groups + DISPATCH_WIDTH - 1u) / DISPATCH_WIDTH;
			occluded.dispatch_z = 1u;
			occluded.index_end = s_running;
		}
	}

	for (uint job_index = lid; job_index < params.job_count; job_index += WORKGROUP_SIZE) {
		uint command = job_index * COMMAND_STRIDE;
		if (draw_commands.data[command + 2u] + draw_commands.data[command] > params.index_capacity) {
			// Out of space: skip this draw. The CPU grows the output buffer for the next frames.
			draw_commands.data[command] = 0u;
			batch.overflow = 1u;
		}
	}
}

#endif // MODE_PREFIX

#ifdef MODE_EMIT

shared uint s_output_offset;

void main() {
	uint entry_index = gl_WorkGroupID.y * DISPATCH_WIDTH + gl_WorkGroupID.x;
	if (entry_index >= min(batch.visible_count, params.visible_capacity)) {
		return;
	}

	uvec2 entry = visible_clusters.data[entry_index];
	uint command = entry.x * COMMAND_STRIDE;
	if (draw_commands.data[command] == 0u) {
		return; // The job didn't fit in the output buffer.
	}

	uint data_offset = jobs.data[entry.x].data_offset;
	uint cluster_index = jobs.data[entry.x].cluster_offset + entry.y;
	uint triangle_count = (clusters.data[cluster_index].counts & 0xFFu) + 1u;
	uint vertex_base = data_offset + clusters.data[cluster_index].vertex_offset;
	uint triangle_base = data_offset + clusters.data[cluster_index].triangle_offset;

	if (gl_LocalInvocationIndex == 0u) {
		s_output_offset = draw_commands.data[command + 2u] + atomicAdd(draw_commands.data[command + 5u], triangle_count * 3u);
	}
	barrier();

	uint triangle = gl_LocalInvocationIndex;
	if (triangle < triangle_count) {
		uint packed_triangle = cluster_data.data[triangle_base + triangle];
		uint output_offset = s_output_offset + triangle * 3u;
		output_indices.data[output_offset + 0u] = cluster_data.data[vertex_base + (packed_triangle & 0xFFu)];
		output_indices.data[output_offset + 1u] = cluster_data.data[vertex_base + ((packed_triangle >> 8u) & 0xFFu)];
		output_indices.data[output_offset + 2u] = cluster_data.data[vertex_base + ((packed_triangle >> 16u) & 0xFFu)];
	}
}

#endif // MODE_EMIT
