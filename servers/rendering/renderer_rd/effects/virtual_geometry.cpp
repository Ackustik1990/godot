/**************************************************************************/
/*  virtual_geometry.cpp                                                  */
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

#include "virtual_geometry.h"

#include "core/config/project_settings.h"
#include "core/object/callable_mp.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/mesh_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/rendering_server_globals.h"

using namespace RendererRD;

VirtualGeometry *VirtualGeometry::singleton = nullptr;

static constexpr uint32_t CULL_WORKGROUP_SIZE = 64;
static constexpr uint32_t DISPATCH_WIDTH = 32768;
static constexpr uint32_t BATCH_STATE_SIZE = 8 * sizeof(uint32_t);
static constexpr uint32_t BATCH_STATE_DISPATCH_OFFSET = 2 * sizeof(uint32_t);
static constexpr uint32_t OCCLUDED_HEADER_SIZE = 8 * sizeof(uint32_t);
static constexpr uint32_t OCCLUDED_DISPATCH_OFFSET = 1 * sizeof(uint32_t);

bool VirtualGeometry::GrowableBuffer::ensure(uint32_t p_size, const char *p_name, bool p_indirect) {
	if (buffer.is_valid() && size >= p_size) {
		return false;
	}
	free();
	size = MAX(Math::next_power_of_2(MAX(p_size, 256u)), p_size);
	buffer = RD::get_singleton()->storage_buffer_create(size, Span<uint8_t>(), p_indirect ? RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT : 0);
	RD::get_singleton()->set_resource_name(buffer, p_name);
	return true;
}

void VirtualGeometry::GrowableBuffer::free() {
	if (buffer.is_valid()) {
		RD::get_singleton()->free_rid(buffer);
		buffer = RID();
	}
	size = 0;
}

void VirtualGeometry::update_settings() {
	enabled = GLOBAL_GET_CACHED(bool, "rendering/virtual_geometry/enabled");
	occlusion_enabled = GLOBAL_GET_CACHED(bool, "rendering/virtual_geometry/occlusion_culling");
}

void VirtualGeometry::_ensure_output_capacity(Batch *p_batch) {
	// Size the output for the peak seen in previous frames plus headroom, but never above the
	// exact worst case of this batch (every job at full detail), so small scenes can't overflow.
	const uint64_t max_indices = MIN(uint64_t(CLAMP(int(GLOBAL_GET_CACHED(int, "rendering/virtual_geometry/max_index_buffer_size")), 1, 1024)) * 1024 * 1024, uint64_t(UINT32_MAX / sizeof(uint32_t)));
	const uint32_t required = MAX(p_batch->passes[PASS_MAIN].required_indices, p_batch->passes[PASS_DISOCCLUDED].required_indices);
	uint64_t target = MAX(uint64_t(required) * 3 / 2, uint64_t(4 * 1024 * 1024));
	target = MIN(target, p_batch->index_upper_bound);
	target = MIN(target, max_indices);
	target = MAX(target, uint64_t(3 * 1024));
	if (p_batch->output_indices.is_valid() && p_batch->output_capacity >= target) {
		return;
	}
	// Round up to avoid reallocating for small increases.
	target = MIN(Math::division_round_up(target, uint64_t(256 * 1024)) * 256 * 1024, max_indices);
	target = MAX(target, uint64_t(p_batch->output_capacity));

	if (p_batch->output_indices.is_valid()) {
		RD::get_singleton()->free_rid(p_batch->output_indices); // Frees the index array too.
	}
	p_batch->output_capacity = uint32_t(target);
	p_batch->output_indices = RD::get_singleton()->index_buffer_create(p_batch->output_capacity, RD::INDEX_BUFFER_FORMAT_UINT32, Span<uint8_t>(), false, RD::BUFFER_CREATION_AS_STORAGE_BIT);
	RD::get_singleton()->set_resource_name(p_batch->output_indices, "VirtualGeometryIndices");
	p_batch->output_index_array = RD::get_singleton()->index_array_create(p_batch->output_indices, 0, p_batch->output_capacity);
}

void VirtualGeometry::_pass_readback(const Vector<uint8_t> &p_data, int p_batch, int p_pass) {
	if (!singleton || p_data.size() < int(BATCH_STATE_SIZE) || p_batch < 0 || p_batch >= int(singleton->batches.size()) || p_pass < 0 || p_pass >= PASS_MAX) {
		return;
	}
	// See BatchState in virtual_geometry.glsl.
	const uint32_t *counters = reinterpret_cast<const uint32_t *>(p_data.ptr());
	const uint32_t index_end = counters[1];
	const uint32_t index_base = MIN(counters[6], index_end);
	Pass &pass = singleton->batches[p_batch]->passes[p_pass];
	pass.last_visible_clusters = counters[0];
	pass.last_index_count = index_end - index_base;
	// Remember the peak, so a moving camera doesn't reallocate every few frames.
	pass.required_indices = MAX(pass.required_indices, index_end);
	if (counters[5] != 0) {
		print_verbose(vformat("Virtual geometry: %d indices didn't fit in the output buffer, growing it.", index_end));
	}
}

void VirtualGeometry::_begin_frame() {
	const uint64_t frame = RSG::rasterizer->get_frame_number();
	if (frame == last_frame) {
		return;
	}
	last_frame = frame;
	batches_used_last_frame = batches_used;
	batches_used = 0;
	last_batch = -1;
}

void VirtualGeometry::begin_batch() {
	ERR_FAIL_COND_MSG(current_batch != -1, "A virtual geometry batch is already being recorded.");
	_begin_frame();

	// Flush pending cluster uploads, so newly loaded meshes can be used right away.
	MeshStorage::get_singleton()->get_virtual_geometry_pool()->flush_uploads();

	if (batches_used == batches.size()) {
		batches.push_back(memnew(Batch));
	}
	current_batch = batches_used++;
	Batch *batch = batches[current_batch];
	batch->job_data.clear();
	batch->view_data.clear();
	batch->view_origins.clear();
	batch->command_data.clear();
	batch->workgroup_count = 0;
	batch->cluster_count = 0;
	batch->index_upper_bound = 0;
	batch->occlusion = false;
}

uint32_t VirtualGeometry::add_view(const ViewParams &p_view) {
	ERR_FAIL_COND_V(current_batch == -1, 0);
	Batch *batch = batches[current_batch];

	// Everything is expressed relative to the culling camera to keep float precision
	// high around it, even with large world coordinates.
	const Vector3 origin = p_view.transform.origin;

	ViewGPU view;
	memset(&view, 0, sizeof(ViewGPU));

	const Vector<Plane> planes = p_view.projection.get_projection_planes(p_view.transform);
	for (int i = 0; i < 6; i++) {
		if (i < planes.size()) {
			const Plane &plane = planes[i];
			view.planes[i][0] = plane.normal.x;
			view.planes[i][1] = plane.normal.y;
			view.planes[i][2] = plane.normal.z;
			view.planes[i][3] = plane.d - plane.normal.dot(origin);
		}
	}
	view.plane_mask = planes.size() == 6 ? p_view.plane_mask : 0;

	const Vector3 lod_position = p_view.lod_position - origin;
	view.lod_position[0] = lod_position.x;
	view.lod_position[1] = lod_position.y;
	view.lod_position[2] = lod_position.z;

	if (p_view.lod_threshold > 0.0f && p_view.lod_distance_multiplier > 0.0f) {
		view.lod_position[3] = 1.0f / (p_view.lod_distance_multiplier * p_view.lod_threshold);
	} else {
		view.flags |= VIEW_FLAG_FORCE_LOD0;
	}

	if (p_view.orthogonal) {
		const Vector3 direction = -p_view.transform.basis.get_column(Vector3::AXIS_Z).normalized();
		view.cull_position[0] = direction.x;
		view.cull_position[1] = direction.y;
		view.cull_position[2] = direction.z;
		view.flags |= VIEW_FLAG_CULL_ORTHOGONAL;
	}
	// For perspective views the camera sits at the origin of the relative space.

	if (p_view.lod_orthogonal) {
		view.flags |= VIEW_FLAG_LOD_ORTHOGONAL;
	}
	if (p_view.frustum_culling) {
		view.flags |= VIEW_FLAG_FRUSTUM;
	}
	if (p_view.cone_culling) {
		view.flags |= VIEW_FLAG_CONE;
	}

	if (p_view.occlusion_culling && occlusion_enabled && p_view.hzb_mip_count > 0) {
		batch->occlusion = true;
		// The matrices take positions relative to the view origin.
		const Projection relative_to_world = Projection(Transform3D(Basis(), origin));
		MaterialStorage::store_camera(p_view.occlusion_previous * relative_to_world, view.occlusion_previous);
		MaterialStorage::store_camera(p_view.occlusion_current * relative_to_world, view.occlusion_current);
		view.hzb_size[0] = p_view.depth_size.width;
		view.hzb_size[1] = p_view.depth_size.height;
		view.hzb_size[2] = p_view.hzb_mip_count;
		if (p_view.occlusion_previous_valid) {
			view.flags |= VIEW_FLAG_OCCLUSION_PREVIOUS;
		}
	}

	batch->view_data.push_back(view);
	batch->view_origins.push_back(origin);
	return batch->view_data.size() - 1;
}

uint32_t VirtualGeometry::add_job(uint32_t p_view, const Transform3D &p_transform, bool p_non_uniform_scale, const VirtualGeometryPool::Allocation &p_allocation, uint32_t p_flags) {
	ERR_FAIL_COND_V(current_batch == -1, INVALID_DRAW);
	Batch *batch = batches[current_batch];
	ERR_FAIL_UNSIGNED_INDEX_V(p_view, batch->view_data.size(), INVALID_DRAW);
	ERR_FAIL_COND_V(!p_allocation.resident || p_allocation.cluster_count == 0, INVALID_DRAW);

	const Vector3 origin = p_transform.origin - batch->view_origins[p_view];
	const Basis &basis = p_transform.basis;

	JobGPU job;
	for (int i = 0; i < 3; i++) {
		job.transform[i * 4 + 0] = basis.rows[i][0];
		job.transform[i * 4 + 1] = basis.rows[i][1];
		job.transform[i * 4 + 2] = basis.rows[i][2];
		job.transform[i * 4 + 3] = origin[i];
	}
	job.cluster_offset = p_allocation.cluster_offset;
	job.cluster_count = p_allocation.cluster_count;
	job.data_offset = p_allocation.data_offset;
	job.workgroup_offset = batch->workgroup_count;
	job.view_index = p_view;
	// Normal cones are only preserved by rotations, mirroring and uniform scaling.
	job.flags = p_non_uniform_scale ? 0 : p_flags;
	job.scale = MAX(basis.get_column(0).length(), MAX(basis.get_column(1).length(), basis.get_column(2).length()));
	job.pad = 0;

	batch->workgroup_count += Math::division_round_up(p_allocation.cluster_count, CULL_WORKGROUP_SIZE);
	batch->cluster_count += p_allocation.cluster_count;
	batch->index_upper_bound += uint64_t(p_allocation.lod0_triangle_count) * 3;
	batch->job_data.push_back(job);

	// VkDrawIndexedIndirectCommand + write cursor + padding.
	const uint32_t command[8] = { 0, 1, 0, 0, 0, 0, 0, 0 };
	for (uint32_t value : command) {
		batch->command_data.push_back(value);
	}

	return batch->job_data.size() - 1;
}

uint32_t VirtualGeometry::get_job_count() const {
	ERR_FAIL_COND_V(current_batch == -1, 0);
	return batches[current_batch]->job_data.size();
}

void VirtualGeometry::_prepare_pass(Batch *p_batch, Pass *p_pass, const char *p_name) {
	RD *rd = RD::get_singleton();

	p_pass->commands.ensure(p_batch->command_data.size() * sizeof(uint32_t), p_name, true);
	if (p_pass->state.is_null()) {
		p_pass->state = rd->storage_buffer_create(BATCH_STATE_SIZE, Span<uint8_t>(), RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
		rd->set_resource_name(p_pass->state, "VirtualGeometryBatchState");
	}

	rd->buffer_update(p_pass->commands.buffer, 0, p_batch->command_data.size() * sizeof(uint32_t), p_batch->command_data.ptr());
	rd->buffer_clear(p_pass->state, 0, BATCH_STATE_SIZE);
}

void VirtualGeometry::_dispatch_pass(Batch *p_batch, int p_batch_index, PassIndex p_pass_index, RID p_hzb) {
	RD *rd = RD::get_singleton();
	VirtualGeometryPool *pool = MeshStorage::get_singleton()->get_virtual_geometry_pool();
	Pass &pass = p_batch->passes[p_pass_index];

	if (p_hzb.is_null()) {
		p_hzb = dummy_hzb_buffer;
	}

	RD::Uniform u_clusters(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, pool->get_cluster_buffer());
	RD::Uniform u_data(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, pool->get_data_buffer());
	RD::Uniform u_jobs(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, p_batch->jobs.buffer);
	RD::Uniform u_views(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, p_batch->views.buffer);
	RD::Uniform u_commands(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, pass.commands.buffer);
	RD::Uniform u_visible(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, p_batch->visible.buffer);
	RD::Uniform u_state(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, pass.state);
	RD::Uniform u_output(RD::UNIFORM_TYPE_STORAGE_BUFFER, 7, p_batch->output_indices);
	RD::Uniform u_occluded(RD::UNIFORM_TYPE_STORAGE_BUFFER, 8, p_batch->occluded.buffer);
	RD::Uniform u_hzb(RD::UNIFORM_TYPE_STORAGE_BUFFER, 9, p_hzb);

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();

	PushConstant push_constant;
	push_constant.job_count = p_batch->job_data.size();
	push_constant.workgroup_count = p_batch->workgroup_count;
	push_constant.index_capacity = p_batch->output_capacity;
	push_constant.visible_capacity = p_batch->cluster_count;
	push_constant.flags = p_pass_index == PASS_DISOCCLUDED ? PUSH_CONSTANT_FLAG_APPEND : 0;
	memset(push_constant.pad, 0, sizeof(push_constant.pad));

	if (p_pass_index == PASS_MAIN) {
		rd->draw_command_begin_label("Virtual Geometry Cull");
	} else {
		rd->draw_command_begin_label("Virtual Geometry Cull Disoccluded");
	}

	const Mode modes[3] = { p_pass_index == PASS_MAIN ? MODE_CULL : MODE_CULL_OCCLUDED, MODE_PREFIX, MODE_EMIT };
	// Each step is its own compute list, so the render graph inserts the barriers between
	// them, including the ones needed to read indirect dispatch arguments.
	for (Mode mode : modes) {
		RID shader_rid = shader.version_get_shader(shader_version, mode);
		RID uniform_set = uniform_set_cache->get_cache(shader_rid, 0, u_clusters, u_data, u_jobs, u_views, u_commands, u_visible, u_state, u_output, u_occluded, u_hzb);

		RD::ComputeListID compute_list = rd->compute_list_begin();
		rd->compute_list_bind_compute_pipeline(compute_list, pipelines[mode]);
		rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
		rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(PushConstant));
		switch (mode) {
			case MODE_CULL: {
				const uint32_t groups = p_batch->workgroup_count;
				rd->compute_list_dispatch(compute_list, MIN(groups, DISPATCH_WIDTH), Math::division_round_up(groups, DISPATCH_WIDTH), 1);
			} break;
			case MODE_CULL_OCCLUDED: {
				rd->compute_list_dispatch_indirect(compute_list, p_batch->occluded.buffer, OCCLUDED_DISPATCH_OFFSET);
			} break;
			case MODE_PREFIX: {
				rd->compute_list_dispatch(compute_list, 1, 1, 1);
			} break;
			case MODE_EMIT: {
				rd->compute_list_dispatch_indirect(compute_list, pass.state, BATCH_STATE_DISPATCH_OFFSET);
			} break;
			default:
				break;
		}
		rd->compute_list_end();
	}

	rd->draw_command_end_label();

	// Read the counters back to size the output buffer of this slot in later frames.
	rd->buffer_get_data_async(pass.state, callable_mp_static(&VirtualGeometry::_pass_readback).bind(p_batch_index, int(p_pass_index)));
}

void VirtualGeometry::end_batch(RID p_hzb_buffer) {
	ERR_FAIL_COND(current_batch == -1);
	const int32_t batch_index = current_batch;
	Batch *batch = batches[batch_index];
	current_batch = -1;

	const uint32_t job_count = batch->job_data.size();
	if (job_count == 0) {
		batch->occlusion = false;
		return;
	}
	last_batch = batch_index;
	batch->last_job_count = job_count;

	VirtualGeometryPool *pool = MeshStorage::get_singleton()->get_virtual_geometry_pool();
	ERR_FAIL_COND(pool->get_cluster_buffer().is_null() || pool->get_data_buffer().is_null());

	RD *rd = RD::get_singleton();

	batch->jobs.ensure(job_count * sizeof(JobGPU), "VirtualGeometryJobs", false);
	batch->views.ensure(batch->view_data.size() * sizeof(ViewGPU), "VirtualGeometryViews", false);
	batch->visible.ensure(batch->cluster_count * sizeof(uint32_t) * 2, "VirtualGeometryVisibleClusters", false);
	batch->occluded.ensure(OCCLUDED_HEADER_SIZE + (batch->occlusion ? batch->cluster_count * sizeof(uint32_t) * 2 : 8), "VirtualGeometryOccludedClusters", true);

	rd->buffer_update(batch->jobs.buffer, 0, job_count * sizeof(JobGPU), batch->job_data.ptr());
	rd->buffer_update(batch->views.buffer, 0, batch->view_data.size() * sizeof(ViewGPU), batch->view_data.ptr());
	rd->buffer_clear(batch->occluded.buffer, 0, OCCLUDED_HEADER_SIZE);

	// Sized once for both passes: the draws of the main pass use it before the second pass is recorded.
	_ensure_output_capacity(batch);
	_prepare_pass(batch, &batch->passes[PASS_MAIN], "VirtualGeometryCommands");
	_dispatch_pass(batch, batch_index, PASS_MAIN, batch->occlusion ? p_hzb_buffer : RID());
}

bool VirtualGeometry::has_disoccluded_pass() const {
	return last_batch >= 0 && batches[last_batch]->occlusion;
}

void VirtualGeometry::process_disoccluded(RID p_hzb_buffer) {
	ERR_FAIL_COND(!has_disoccluded_pass());
	ERR_FAIL_COND(p_hzb_buffer.is_null());
	Batch *batch = batches[last_batch];
	_prepare_pass(batch, &batch->passes[PASS_DISOCCLUDED], "VirtualGeometryDisoccludedCommands");
	_dispatch_pass(batch, last_batch, PASS_DISOCCLUDED, p_hzb_buffer);
}

uint32_t VirtualGeometry::get_hzb_mip_count(const Size2i &p_depth_size) {
	// Same as a regular mipmap chain: floor(log2(max(width, height))) + 1.
	const Size2i size = get_hzb_size(p_depth_size);
	uint32_t count = 1;
	uint32_t dimension = MAX(size.width, size.height);
	while (dimension > 1) {
		dimension >>= 1;
		count++;
	}
	return count;
}

uint32_t VirtualGeometry::get_hzb_buffer_size(const Size2i &p_depth_size) {
	uint64_t texels = 0;
	Size2i size = get_hzb_size(p_depth_size);
	const uint32_t mip_count = get_hzb_mip_count(p_depth_size);
	for (uint32_t i = 0; i < mip_count; i++) {
		texels += uint64_t(size.width) * size.height;
		size = Size2i(MAX(size.width / 2, 1), MAX(size.height / 2, 1));
	}
	return uint32_t(texels * sizeof(float));
}

void VirtualGeometry::build_hzb(RID p_depth, const Size2i &p_depth_size, RID p_hzb_buffer) {
	ERR_FAIL_COND(p_hzb_buffer.is_null());
	RD *rd = RD::get_singleton();
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	const RID sampler = MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	rd->draw_command_begin_label("Virtual Geometry HZB");

	const uint32_t mip_count = get_hzb_mip_count(p_depth_size);
	const RD::Uniform u_hzb(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, p_hzb_buffer);
	Size2i source_size = p_depth_size;
	uint32_t source_offset = 0;
	uint32_t dest_offset = 0;
	for (uint32_t i = 0; i < mip_count; i++) {
		const Size2i dest_size(MAX(source_size.width / 2, 1), MAX(source_size.height / 2, 1));
		const HzbMode mode = i == 0 ? HZB_MODE_DEPTH : HZB_MODE_REDUCE;
		RID shader_rid = hzb_shader.version_get_shader(hzb_shader_version, mode);

		RID uniform_set;
		if (mode == HZB_MODE_DEPTH) {
			RD::Uniform u_source(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ sampler, p_depth }));
			uniform_set = uniform_set_cache->get_cache(shader_rid, 0, u_source, u_hzb);
		} else {
			uniform_set = uniform_set_cache->get_cache(shader_rid, 0, u_hzb);
		}

		HzbPushConstant push_constant;
		push_constant.source_size[0] = source_size.width;
		push_constant.source_size[1] = source_size.height;
		push_constant.dest_size[0] = dest_size.width;
		push_constant.dest_size[1] = dest_size.height;
		push_constant.source_offset = source_offset;
		push_constant.dest_offset = dest_offset;
		push_constant.pad[0] = 0;
		push_constant.pad[1] = 0;

		// One compute list per level, so the render graph orders the reads after the writes.
		RD::ComputeListID compute_list = rd->compute_list_begin();
		rd->compute_list_bind_compute_pipeline(compute_list, hzb_pipelines[mode]);
		rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
		rd->compute_list_set_push_constant(compute_list, &push_constant, sizeof(HzbPushConstant));
		rd->compute_list_dispatch_threads(compute_list, dest_size.width, dest_size.height, 1);
		rd->compute_list_end();

		source_size = dest_size;
		source_offset = dest_offset;
		dest_offset += dest_size.width * dest_size.height;
	}

	rd->draw_command_end_label();
}

RID VirtualGeometry::get_command_buffer(PassIndex p_pass) const {
	ERR_FAIL_COND_V(last_batch < 0, RID());
	return batches[last_batch]->passes[p_pass].commands.buffer;
}

RID VirtualGeometry::get_index_array() const {
	ERR_FAIL_COND_V(last_batch < 0, RID());
	return batches[last_batch]->output_index_array;
}

VirtualGeometry::Stats VirtualGeometry::get_stats() const {
	Stats stats;
	const uint32_t count = MAX(batches_used, batches_used_last_frame);
	for (uint32_t i = 0; i < count && i < batches.size(); i++) {
		const Batch *batch = batches[i];
		stats.jobs += batch->last_job_count;
		stats.index_capacity += batch->output_capacity;
		for (const Pass &pass : batch->passes) {
			stats.visible_clusters += pass.last_visible_clusters;
			stats.triangles += MIN(pass.last_index_count, batch->output_capacity) / 3;
		}
	}
	return stats;
}

VirtualGeometry::VirtualGeometry() {
	singleton = this;

	{
		Vector<String> modes;
		modes.push_back("\n#define MODE_CULL\n");
		modes.push_back("\n#define MODE_CULL_OCCLUDED\n");
		modes.push_back("\n#define MODE_PREFIX\n");
		modes.push_back("\n#define MODE_EMIT\n");
		shader.initialize(modes);
		shader_version = shader.version_create();
		for (int i = 0; i < MODE_MAX; i++) {
			pipelines[i] = RD::get_singleton()->compute_pipeline_create(shader.version_get_shader(shader_version, i));
		}
	}

	{
		Vector<String> modes;
		modes.push_back("\n#define MODE_DEPTH\n");
		modes.push_back("\n#define MODE_REDUCE\n");
		hzb_shader.initialize(modes);
		hzb_shader_version = hzb_shader.version_create();
		for (int i = 0; i < HZB_MODE_MAX; i++) {
			hzb_pipelines[i] = RD::get_singleton()->compute_pipeline_create(hzb_shader.version_get_shader(hzb_shader_version, i));
		}
		dummy_hzb_buffer = RD::get_singleton()->storage_buffer_create(16);
		RD::get_singleton()->set_resource_name(dummy_hzb_buffer, "VirtualGeometryDummyHZB");
	}

	update_settings();
}

VirtualGeometry::~VirtualGeometry() {
	for (Batch *batch : batches) {
		batch->jobs.free();
		batch->views.free();
		batch->visible.free();
		batch->occluded.free();
		for (Pass &pass : batch->passes) {
			pass.commands.free();
			if (pass.state.is_valid()) {
				RD::get_singleton()->free_rid(pass.state);
			}
		}
		if (batch->output_indices.is_valid()) {
			RD::get_singleton()->free_rid(batch->output_indices);
		}
		memdelete(batch);
	}
	batches.clear();

	RD::get_singleton()->free_rid(dummy_hzb_buffer);
	shader.version_free(shader_version);
	hzb_shader.version_free(hzb_shader_version);
	singleton = nullptr;
}
