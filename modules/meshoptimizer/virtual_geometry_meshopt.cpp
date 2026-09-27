/**************************************************************************/
/*  virtual_geometry_meshopt.cpp                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "virtual_geometry_meshopt.h"

#include "core/object/worker_thread_pool.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/virtual_geometry_format.h"

#include <thirdparty/meshoptimizer/meshoptimizer.h>

#include <cfloat>

// Cluster DAG construction.
//
// 1. The surface is split into clusters of up to 128 triangles.
// 2. Clusters are partitioned into groups of spatially adjacent clusters.
// 3. Each group is merged and simplified to half of its triangles while its
//    border with other groups is locked, then split again into new clusters.
//    Locking the border guarantees that any combination of neighboring groups
//    at different levels of detail is crack-free.
// 4. Steps 2-3 repeat on the new clusters until the mesh can't be reduced.
//
// Every cluster stores the bounds and error of the group it was generated from
// ("self") and of the group it was simplified into ("parent"). Group bounds
// enclose the bounds of their children and group errors are monotonic, so the
// runtime test "self error is small enough and parent error is too large"
// selects a consistent, watertight cut through the DAG independently for each
// cluster, which is what makes a fully parallel GPU selection possible.

namespace {

using VirtualGeometryFormat::Cluster;
using VirtualGeometryFormat::Header;

struct LODBounds {
	float center[3] = { 0.0f, 0.0f, 0.0f };
	float radius = 0.0f;
	float error = 0.0f;
};

struct BuildCluster {
	LocalVector<uint32_t> indices;
	LODBounds self;
	LODBounds parent;
	uint32_t level = 0;
	bool has_parent = false;
};

struct BuildContext {
	const float *positions = nullptr;
	const float *normals = nullptr;
	uint32_t vertex_count = 0;
	VirtualGeometryBuilder::Settings settings;
};

struct GroupTask {
	LocalVector<uint32_t> members;
	LocalVector<LocalVector<uint32_t>> new_clusters;
	LODBounds bounds;
	bool success = false;
};

struct GroupTaskData {
	const BuildContext *context = nullptr;
	const LocalVector<BuildCluster> *clusters = nullptr;
	GroupTask *tasks = nullptr;
};

constexpr float CONE_WEIGHT = 0.25f;
// Groups that can't be reduced to this fraction of their triangles are retried at the next level.
constexpr float MIN_REDUCTION = 0.85f;
constexpr uint32_t MAX_LEVELS = 32;

void clusterize(const BuildContext &p_context, const uint32_t *p_indices, size_t p_index_count, LocalVector<LocalVector<uint32_t>> &r_clusters) {
	const size_t max_vertices = p_context.settings.max_cluster_vertices;
	const size_t max_triangles = p_context.settings.max_cluster_triangles;

	LocalVector<meshopt_Meshlet> meshlets;
	meshlets.resize(meshopt_buildMeshletsBound(p_index_count, max_vertices, max_triangles));
	LocalVector<uint32_t> meshlet_vertices;
	meshlet_vertices.resize(p_index_count);
	LocalVector<uint8_t> meshlet_triangles;
	meshlet_triangles.resize(p_index_count);

	const size_t meshlet_count = meshopt_buildMeshlets(meshlets.ptr(), meshlet_vertices.ptr(), meshlet_triangles.ptr(), p_indices, p_index_count, p_context.positions, p_context.vertex_count, sizeof(float) * 3, max_vertices, max_triangles, CONE_WEIGHT);

	const uint32_t first = r_clusters.size();
	r_clusters.resize(first + meshlet_count);
	for (size_t i = 0; i < meshlet_count; i++) {
		const meshopt_Meshlet &meshlet = meshlets[i];
		LocalVector<uint32_t> &cluster = r_clusters[first + i];
		cluster.resize(meshlet.triangle_count * 3);
		for (uint32_t j = 0; j < meshlet.triangle_count * 3; j++) {
			cluster[j] = meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j]];
		}
	}
}

LODBounds compute_cluster_sphere(const BuildContext &p_context, const LocalVector<uint32_t> &p_indices) {
	const meshopt_Bounds bounds = meshopt_computeClusterBounds(p_indices.ptr(), p_indices.size(), p_context.positions, p_context.vertex_count, sizeof(float) * 3);
	LODBounds result;
	result.center[0] = bounds.center[0];
	result.center[1] = bounds.center[1];
	result.center[2] = bounds.center[2];
	result.radius = bounds.radius;
	result.error = 0.0f;
	return result;
}

// Merges the "self" bounds of all clusters of a group. The resulting sphere encloses
// every child sphere and the error is the largest child error, which keeps both
// monotonic along the DAG.
LODBounds merge_group_bounds(const LocalVector<BuildCluster> &p_clusters, const LocalVector<uint32_t> &p_members) {
	LocalVector<float> centers;
	LocalVector<float> radii;
	centers.resize(p_members.size() * 3);
	radii.resize(p_members.size());

	LODBounds result;
	for (uint32_t i = 0; i < p_members.size(); i++) {
		const LODBounds &child = p_clusters[p_members[i]].self;
		centers[i * 3 + 0] = child.center[0];
		centers[i * 3 + 1] = child.center[1];
		centers[i * 3 + 2] = child.center[2];
		radii[i] = child.radius;
		result.error = MAX(result.error, child.error);
	}

	const meshopt_Bounds sphere = meshopt_computeSphereBounds(centers.ptr(), p_members.size(), sizeof(float) * 3, radii.ptr(), sizeof(float));
	result.center[0] = sphere.center[0];
	result.center[1] = sphere.center[1];
	result.center[2] = sphere.center[2];
	result.radius = sphere.radius;

	// Guard against floating point imprecision: the parent sphere must contain every child sphere.
	for (uint32_t i = 0; i < p_members.size(); i++) {
		const float dx = centers[i * 3 + 0] - result.center[0];
		const float dy = centers[i * 3 + 1] - result.center[1];
		const float dz = centers[i * 3 + 2] - result.center[2];
		const float needed = Math::sqrt(dx * dx + dy * dy + dz * dz) + radii[i];
		result.radius = MAX(result.radius, needed);
	}
	return result;
}

void simplify_group(void *p_userdata, uint32_t p_index) {
	GroupTaskData *data = static_cast<GroupTaskData *>(p_userdata);
	const BuildContext &context = *data->context;
	const LocalVector<BuildCluster> &clusters = *data->clusters;
	GroupTask &task = data->tasks[p_index];

	task.success = false;
	if (task.members.size() < 2) {
		// A lone cluster can't be simplified meaningfully with its border locked; retry it with other neighbors.
		return;
	}

	LocalVector<uint32_t> merged;
	for (uint32_t member : task.members) {
		const LocalVector<uint32_t> &indices = clusters[member].indices;
		for (uint32_t index : indices) {
			merged.push_back(index);
		}
	}

	const size_t target_index_count = (merged.size() / 3 / 2) * 3;
	LocalVector<uint32_t> simplified;
	simplified.resize(merged.size());

	const unsigned int options = meshopt_SimplifyLockBorder | meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute;
	float error = 0.0f;
	size_t simplified_count = 0;
	if (context.normals && context.settings.normal_weight > 0.0f) {
		const float weights[3] = { context.settings.normal_weight, context.settings.normal_weight, context.settings.normal_weight };
		simplified_count = meshopt_simplifyWithAttributes(simplified.ptr(), merged.ptr(), merged.size(), context.positions, context.vertex_count, sizeof(float) * 3, context.normals, sizeof(float) * 3, weights, 3, nullptr, target_index_count, FLT_MAX, options, &error);
	} else {
		simplified_count = meshopt_simplify(simplified.ptr(), merged.ptr(), merged.size(), context.positions, context.vertex_count, sizeof(float) * 3, target_index_count, FLT_MAX, options, &error);
	}

	if (simplified_count == 0 || float(simplified_count) > float(merged.size()) * MIN_REDUCTION) {
		return;
	}

	task.bounds = merge_group_bounds(clusters, task.members);
	// Errors accumulate: the simplified group deviates from its children by `error`,
	// and the children deviate from the original surface by up to their own error.
	task.bounds.error += error;

	task.new_clusters.clear();
	clusterize(context, simplified.ptr(), simplified_count, task.new_clusters);
	task.success = !task.new_clusters.is_empty();
}

uint32_t count_triangles(const LocalVector<BuildCluster> &p_clusters, const LocalVector<uint32_t> &p_ids) {
	uint32_t count = 0;
	for (uint32_t id : p_ids) {
		count += p_clusters[id].indices.size() / 3;
	}
	return count;
}

uint32_t pack_snorm8(int32_t p_value) {
	return uint32_t(CLAMP(p_value, -127, 127)) & 0xFF;
}

} // namespace

Vector<uint8_t> virtual_geometry_build_meshopt(const float *p_positions, const float *p_normals, uint32_t p_vertex_count, const uint32_t *p_indices, uint32_t p_index_count, const VirtualGeometryBuilder::Settings &p_settings) {
	ERR_FAIL_NULL_V(p_positions, Vector<uint8_t>());
	ERR_FAIL_NULL_V(p_indices, Vector<uint8_t>());
	ERR_FAIL_COND_V(p_index_count < 3 || p_index_count % 3 != 0, Vector<uint8_t>());

	BuildContext context;
	context.positions = p_positions;
	context.normals = p_normals;
	context.vertex_count = p_vertex_count;
	context.settings = p_settings;
	context.settings.max_cluster_vertices = CLAMP(context.settings.max_cluster_vertices, 3u, VirtualGeometryFormat::MAX_CLUSTER_VERTICES);
	context.settings.max_cluster_triangles = CLAMP(context.settings.max_cluster_triangles, 1u, VirtualGeometryFormat::MAX_CLUSTER_TRIANGLES);
	context.settings.group_size = CLAMP(context.settings.group_size, 2u, 64u);

	LocalVector<BuildCluster> clusters;
	LocalVector<uint32_t> pending;

	// Level 0: the original surface.
	{
		LocalVector<LocalVector<uint32_t>> level0;
		clusterize(context, p_indices, p_index_count, level0);
		ERR_FAIL_COND_V_MSG(level0.is_empty(), Vector<uint8_t>(), "Failed to split the surface into clusters.");
		clusters.resize(level0.size());
		for (uint32_t i = 0; i < level0.size(); i++) {
			BuildCluster &cluster = clusters[i];
			cluster.indices = std::move(level0[i]);
			cluster.self = compute_cluster_sphere(context, cluster.indices);
			cluster.level = 0;
			pending.push_back(i);
		}
	}

	uint32_t level = 0;
	while (pending.size() > 1 && level + 1 < MAX_LEVELS) {
		// Group clusters that share vertices or are close to each other.
		LocalVector<uint32_t> partition_indices;
		LocalVector<uint32_t> partition_counts;
		partition_counts.resize(pending.size());
		for (uint32_t i = 0; i < pending.size(); i++) {
			const LocalVector<uint32_t> &indices = clusters[pending[i]].indices;
			for (uint32_t index : indices) {
				partition_indices.push_back(index);
			}
			partition_counts[i] = indices.size();
		}

		LocalVector<uint32_t> partition;
		partition.resize(pending.size());
		const size_t group_count = meshopt_partitionClusters(partition.ptr(), partition_indices.ptr(), partition_indices.size(), partition_counts.ptr(), pending.size(), context.positions, context.vertex_count, sizeof(float) * 3, context.settings.group_size);

		LocalVector<GroupTask> tasks;
		tasks.resize(group_count);
		for (uint32_t i = 0; i < pending.size(); i++) {
			tasks[partition[i]].members.push_back(pending[i]);
		}

		GroupTaskData task_data;
		task_data.context = &context;
		task_data.clusters = &clusters;
		task_data.tasks = tasks.ptr();

		if (context.settings.use_threads && group_count > 1 && WorkerThreadPool::get_singleton()) {
			WorkerThreadPool::GroupID group = WorkerThreadPool::get_singleton()->add_native_group_task(&simplify_group, &task_data, group_count, -1, true, SNAME("VirtualGeometryBuild"));
			WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group);
		} else {
			for (uint32_t i = 0; i < group_count; i++) {
				simplify_group(&task_data, i);
			}
		}

		LocalVector<uint32_t> next;
		for (GroupTask &task : tasks) {
			if (!task.success) {
				for (uint32_t member : task.members) {
					next.push_back(member);
				}
				continue;
			}
			for (uint32_t member : task.members) {
				clusters[member].parent = task.bounds;
				clusters[member].has_parent = true;
			}
			for (LocalVector<uint32_t> &new_cluster : task.new_clusters) {
				BuildCluster cluster;
				cluster.indices = std::move(new_cluster);
				cluster.self = task.bounds;
				cluster.level = level + 1;
				next.push_back(clusters.size());
				clusters.push_back(std::move(cluster));
			}
		}

		const uint32_t pending_triangles = count_triangles(clusters, pending);
		const uint32_t next_triangles = count_triangles(clusters, next);
		pending = std::move(next);
		level++;
		if (float(next_triangles) > float(pending_triangles) * 0.95f) {
			break; // Not enough progress, stop here.
		}
	}

	// Whatever couldn't be simplified further is a root: it is never replaced.
	for (BuildCluster &cluster : clusters) {
		if (!cluster.has_parent) {
			cluster.parent = cluster.self;
			cluster.parent.error = VirtualGeometryFormat::ROOT_ERROR;
			cluster.has_parent = true;
		}
	}

	// Serialize.
	Header header;
	header.cluster_count = clusters.size();
	header.vertex_count = p_vertex_count;

	LocalVector<Cluster> out_clusters;
	out_clusters.resize(clusters.size());
	LocalVector<uint32_t> out_data;
	out_data.reserve(p_index_count);

	uint32_t max_level = 0;
	uint32_t local_vertices[256];
	LocalVector<uint8_t> local_triangles;
	local_triangles.resize(VirtualGeometryFormat::MAX_CLUSTER_TRIANGLES * 3);

	for (uint32_t i = 0; i < clusters.size(); i++) {
		const BuildCluster &src = clusters[i];
		Cluster &dst = out_clusters[i];
		memset(&dst, 0, sizeof(Cluster));

		const uint32_t triangle_count = src.indices.size() / 3;
		ERR_FAIL_COND_V(triangle_count == 0 || triangle_count > VirtualGeometryFormat::MAX_CLUSTER_TRIANGLES, Vector<uint8_t>());

		const uint32_t vertex_count = meshopt_extractMeshletIndices(local_vertices, local_triangles.ptr(), src.indices.ptr(), src.indices.size());
		ERR_FAIL_COND_V(vertex_count == 0 || vertex_count > VirtualGeometryFormat::MAX_CLUSTER_VERTICES, Vector<uint8_t>());
		meshopt_optimizeMeshlet(local_vertices, local_triangles.ptr(), triangle_count, vertex_count);

		const meshopt_Bounds bounds = meshopt_computeMeshletBounds(local_vertices, local_triangles.ptr(), triangle_count, context.positions, context.vertex_count, sizeof(float) * 3);

		dst.bounds[0] = bounds.center[0];
		dst.bounds[1] = bounds.center[1];
		dst.bounds[2] = bounds.center[2];
		dst.bounds[3] = bounds.radius;

		dst.lod_sphere[0] = src.self.center[0];
		dst.lod_sphere[1] = src.self.center[1];
		dst.lod_sphere[2] = src.self.center[2];
		dst.lod_sphere[3] = src.self.radius;
		dst.lod_error = src.self.error;

		dst.parent_sphere[0] = src.parent.center[0];
		dst.parent_sphere[1] = src.parent.center[1];
		dst.parent_sphere[2] = src.parent.center[2];
		dst.parent_sphere[3] = src.parent.radius;
		dst.parent_error = src.parent.error;

		// meshoptimizer derives normals from counter-clockwise winding, while Godot
		// uses clockwise winding for front faces: flip the cone axis.
		if (bounds.cone_cutoff_s8 >= 127) {
			dst.cone = 127u << 24; // Degenerate cone: never cull.
		} else {
			// The quantized cutoff is conservative for the quantized axis, so use both as-is.
			dst.cone = pack_snorm8(-bounds.cone_axis_s8[0]) | (pack_snorm8(-bounds.cone_axis_s8[1]) << 8) | (pack_snorm8(-bounds.cone_axis_s8[2]) << 16) | (pack_snorm8(bounds.cone_cutoff_s8) << 24);
		}

		dst.counts = ((triangle_count - 1) << VirtualGeometryFormat::CLUSTER_TRIANGLE_COUNT_SHIFT) | ((vertex_count - 1) << VirtualGeometryFormat::CLUSTER_VERTEX_COUNT_SHIFT) | (src.level << VirtualGeometryFormat::CLUSTER_LOD_LEVEL_SHIFT);

		dst.vertex_offset = out_data.size();
		for (uint32_t v = 0; v < vertex_count; v++) {
			out_data.push_back(local_vertices[v]);
		}
		dst.triangle_offset = out_data.size();
		for (uint32_t t = 0; t < triangle_count; t++) {
			out_data.push_back(uint32_t(local_triangles[t * 3 + 0]) | (uint32_t(local_triangles[t * 3 + 1]) << 8) | (uint32_t(local_triangles[t * 3 + 2]) << 16));
		}

		max_level = MAX(max_level, src.level);
		if (src.level == 0) {
			header.lod0_triangle_count += triangle_count;
		}
		if (src.parent.error >= VirtualGeometryFormat::ROOT_ERROR) {
			header.root_triangle_count += triangle_count;
		}
	}

	header.data_count = out_data.size();
	header.lod_level_count = max_level + 1;

	Vector<uint8_t> result;
	const uint64_t size = sizeof(Header) + uint64_t(out_clusters.size()) * sizeof(Cluster) + uint64_t(out_data.size()) * sizeof(uint32_t);
	ERR_FAIL_COND_V_MSG(size > uint64_t(INT32_MAX), Vector<uint8_t>(), "Virtual geometry data is too large.");
	result.resize(size);
	uint8_t *w = result.ptrw();
	memcpy(w, &header, sizeof(Header));
	memcpy(w + sizeof(Header), out_clusters.ptr(), out_clusters.size() * sizeof(Cluster));
	memcpy(w + sizeof(Header) + out_clusters.size() * sizeof(Cluster), out_data.ptr(), out_data.size() * sizeof(uint32_t));
	return result;
}
