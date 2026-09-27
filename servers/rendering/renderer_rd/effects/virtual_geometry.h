/**************************************************************************/
/*  virtual_geometry.h                                                    */
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

#pragma once

#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/renderer_rd/shaders/effects/virtual_geometry.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/virtual_geometry_pool.h"

namespace RendererRD {

// GPU-driven cluster LOD renderer for surfaces with virtual geometry.
//
// Work is organized in batches. A batch holds a set of views (a camera, or all the
// shadow passes rendered together) and a set of jobs (one per surface instance and
// view). Processing a batch runs three compute passes that select the clusters of
// each job, cull them, and compact their triangles into a shared index buffer.
// Each job then renders with a single indexed indirect draw, using the regular
// material pipeline and the regular vertex buffers of the surface.
class VirtualGeometry {
public:
	static constexpr uint32_t COMMAND_STRIDE_BYTES = 8 * sizeof(uint32_t);
	static constexpr uint32_t INVALID_DRAW = UINT32_MAX;

	enum JobFlags {
		JOB_FLAG_CONE_CULL = 1, // The material culls back faces: clusters facing away can be skipped.
		JOB_FLAG_CONE_INVERT = 2, // The material culls front faces instead.
	};

	struct ViewParams {
		// Culling camera.
		Projection projection;
		Transform3D transform; // Camera to world.
		bool orthogonal = false;
		bool frustum_culling = true;
		bool cone_culling = true;
		uint32_t plane_mask = 0x3F; // Bits follow Projection::Planes.

		// LOD camera. Shadow views use the main camera so the geometry that casts
		// shadows matches the geometry that is seen.
		Vector3 lod_position;
		bool lod_orthogonal = false;
		float lod_distance_multiplier = 1.0; // Projection::get_lod_multiplier() of the LOD camera.
		float lod_threshold = 0.0; // Screen fraction, see RenderSceneData::screen_mesh_lod_threshold. 0 forces full detail.
	};

	struct Stats {
		uint32_t jobs = 0;
		uint32_t visible_clusters = 0;
		uint32_t triangles = 0;
		uint32_t index_capacity = 0;
	};

private:
	struct JobGPU {
		float transform[12];
		uint32_t cluster_offset;
		uint32_t cluster_count;
		uint32_t data_offset;
		uint32_t workgroup_offset;
		uint32_t view_index;
		uint32_t flags;
		float scale;
		uint32_t pad;
	};
	static_assert(sizeof(JobGPU) == 80);

	struct ViewGPU {
		float planes[6][4];
		float lod_position[4];
		float cull_position[4];
		uint32_t flags;
		uint32_t plane_mask;
		uint32_t pad[2];
	};
	static_assert(sizeof(ViewGPU) == 144);

	enum ViewFlags {
		VIEW_FLAG_LOD_ORTHOGONAL = 1,
		VIEW_FLAG_CULL_ORTHOGONAL = 2,
		VIEW_FLAG_FRUSTUM = 4,
		VIEW_FLAG_CONE = 8,
		VIEW_FLAG_FORCE_LOD0 = 16,
	};

	enum Mode {
		MODE_CULL,
		MODE_PREFIX,
		MODE_EMIT,
		MODE_MAX
	};

	struct PushConstant {
		uint32_t job_count;
		uint32_t workgroup_count;
		uint32_t index_capacity;
		uint32_t visible_capacity;
	};

	struct GrowableBuffer {
		RID buffer;
		uint32_t size = 0;
		bool ensure(uint32_t p_size, const char *p_name, bool p_indirect);
		void free();
	};

	struct Batch {
		GrowableBuffer jobs;
		GrowableBuffer views;
		GrowableBuffer commands;
		GrowableBuffer visible;
		RID state;

		// Index buffer the emit pass writes to. Each batch has its own, so it can be
		// resized before recording without invalidating draws of other batches.
		RID output_indices;
		RID output_index_array;
		uint32_t output_capacity = 0;

		// Filled asynchronously from the GPU counters of previous frames.
		uint32_t required_indices = 0;
		uint32_t last_index_count = 0;
		uint32_t last_visible_clusters = 0;
		uint32_t last_job_count = 0;

		LocalVector<JobGPU> job_data;
		LocalVector<ViewGPU> view_data;
		LocalVector<Vector3> view_origins;
		LocalVector<uint32_t> command_data;
		uint32_t workgroup_count = 0;
		uint32_t cluster_count = 0;
		uint64_t index_upper_bound = 0;
	};

	static VirtualGeometry *singleton;

	VirtualGeometryShaderRD shader;
	RID shader_version;
	RID pipelines[MODE_MAX];

	LocalVector<Batch *> batches;
	uint32_t batches_used = 0;
	uint32_t batches_used_last_frame = 0;
	int32_t current_batch = -1;
	int32_t last_batch = -1;
	uint64_t last_frame = UINT64_MAX;

	bool enabled = true;

	void _ensure_output_capacity(Batch *p_batch, uint32_t p_capacity);
	void _begin_frame();
	static void _batch_readback(const Vector<uint8_t> &p_data, int p_batch);

public:
	static VirtualGeometry *get_singleton() { return singleton; }

	bool is_enabled() const { return enabled; }
	void update_settings();

	// Starts recording a new batch. Only one batch can be recorded at a time.
	void begin_batch();
	uint32_t add_view(const ViewParams &p_view);
	// Returns the draw index of the job, used with get_command_buffer().
	uint32_t add_job(uint32_t p_view, const Transform3D &p_transform, bool p_non_uniform_scale, const VirtualGeometryPool::Allocation &p_allocation, uint32_t p_flags);
	uint32_t get_job_count() const;
	// Records the compute passes. Must be called outside of draw and compute lists.
	void end_batch();

	// Buffers of the last batch processed by end_batch(). They stay valid until the same
	// batch slot is recorded again in a later frame.
	RID get_command_buffer() const;
	RID get_index_array() const;

	// Totals of the batches of a recent frame (read back asynchronously).
	Stats get_stats() const;

	VirtualGeometry();
	~VirtualGeometry();
};

} // namespace RendererRD
