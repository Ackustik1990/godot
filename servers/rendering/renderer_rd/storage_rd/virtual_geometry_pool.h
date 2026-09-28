/**************************************************************************/
/*  virtual_geometry_pool.h                                               */
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

#include "core/os/mutex.h"
#include "core/templates/local_vector.h"
#include "core/templates/rb_map.h"
#include "servers/rendering/rendering_device.h"

namespace RendererRD {

// Stores the cluster hierarchies of every virtual geometry surface in two large
// GPU buffers (one for cluster headers, one for vertex references and packed
// triangles), so a single compute dispatch can process every instance of every
// mesh in a view without rebinding resources.
//
// Uploads are deferred to the render thread: surfaces may be created from
// loader threads, but writing into shared buffers must be recorded by the
// render thread.
class VirtualGeometryPool {
public:
	struct Allocation {
		uint32_t cluster_offset = 0; // In clusters.
		uint32_t cluster_count = 0;
		uint32_t data_offset = 0; // In 32-bit words.
		uint32_t data_count = 0;
		uint32_t vertex_count = 0;
		uint32_t lod0_triangle_count = 0;
		uint32_t root_triangle_count = 0;
		uint32_t lod_level_count = 0;
		bool resident = false; // True once uploaded to the GPU.
	};

private:
	class Heap {
		const char *name = "";
		uint32_t element_size = 0;
		uint32_t capacity = 0; // In elements.
		RID buffer;
		RBMap<uint32_t, uint32_t> free_ranges; // Offset -> size.

		bool _grow(uint32_t p_min_free);

	public:
		bool allocate(uint32_t p_count, uint32_t &r_offset);
		void free(uint32_t p_offset, uint32_t p_count);
		RID get_buffer() const { return buffer; }
		uint32_t get_capacity() const { return capacity; }
		uint32_t get_used() const;
		void finalize();

		Heap(const char *p_name, uint32_t p_element_size) :
				name(p_name), element_size(p_element_size) {}
	};

	struct PendingUpload {
		Allocation *allocation = nullptr;
		Vector<uint8_t> data;
	};

	Heap clusters = Heap("VirtualGeometryClusters", 80);
	Heap data = Heap("VirtualGeometryData", 4);

	Mutex mutex;
	LocalVector<PendingUpload> pending;
	uint64_t version = 1;

public:
	// Queues the data for upload; the allocation becomes resident on the next flush.
	// p_allocation must stay valid until it's released with free().
	void queue_upload(Allocation *p_allocation, const Vector<uint8_t> &p_data);
	void free(Allocation *p_allocation);

	// Must be called from the render thread, outside of draw and compute lists.
	void flush_uploads();

	// Reconstructs the blob of an allocation (synchronous GPU readback).
	Vector<uint8_t> read(const Allocation *p_allocation) const;

	RID get_cluster_buffer() const { return clusters.get_buffer(); }
	RID get_data_buffer() const { return data.get_buffer(); }
	uint64_t get_version() const { return version; }
	uint64_t get_memory_usage() const;

	void finalize();
};

} // namespace RendererRD
