/**************************************************************************/
/*  virtual_geometry_pool.cpp                                             */
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

#include "virtual_geometry_pool.h"

#include "servers/rendering/virtual_geometry_format.h"

using namespace RendererRD;

static constexpr uint32_t VIRTUAL_GEOMETRY_POOL_MIN_ELEMENTS = 4096;

/* Heap */

bool VirtualGeometryPool::Heap::_grow(uint32_t p_min_free) {
	uint64_t new_capacity = MAX(uint64_t(capacity) * 2, uint64_t(capacity) + p_min_free);
	new_capacity = MAX(new_capacity, uint64_t(VIRTUAL_GEOMETRY_POOL_MIN_ELEMENTS));
	const uint64_t max_bytes = uint64_t(UINT32_MAX) & ~uint64_t(0xFFFF);
	if (new_capacity * element_size > max_bytes) {
		new_capacity = max_bytes / element_size;
	}
	ERR_FAIL_COND_V_MSG(new_capacity < uint64_t(capacity) + p_min_free, false, vformat("Virtual geometry pool '%s' is full.", name));

	RID new_buffer = RD::get_singleton()->storage_buffer_create(new_capacity * element_size);
	ERR_FAIL_COND_V(new_buffer.is_null(), false);
	RD::get_singleton()->set_resource_name(new_buffer, name);

	if (buffer.is_valid()) {
		RD::get_singleton()->buffer_copy(buffer, new_buffer, 0, 0, capacity * element_size);
		RD::get_singleton()->free_rid(buffer);
	}

	// Append the new space, merging it with a trailing free range if there is one.
	uint32_t range_offset = capacity;
	uint32_t range_size = uint32_t(new_capacity) - capacity;
	RBMap<uint32_t, uint32_t>::Element *last = free_ranges.back();
	if (last && last->key() + last->get() == capacity) {
		range_offset = last->key();
		range_size += last->get();
		free_ranges.erase(last);
	}
	free_ranges.insert(range_offset, range_size);

	buffer = new_buffer;
	capacity = uint32_t(new_capacity);
	return true;
}

bool VirtualGeometryPool::Heap::allocate(uint32_t p_count, uint32_t &r_offset) {
	ERR_FAIL_COND_V(p_count == 0, false);

	for (int attempt = 0; attempt < 2; attempt++) {
		// First fit. Allocations happen on load, so this doesn't need to be fancy.
		for (RBMap<uint32_t, uint32_t>::Element *E = free_ranges.front(); E; E = E->next()) {
			if (E->get() >= p_count) {
				r_offset = E->key();
				const uint32_t remaining = E->get() - p_count;
				free_ranges.erase(E);
				if (remaining > 0) {
					free_ranges.insert(r_offset + p_count, remaining);
				}
				return true;
			}
		}
		if (!_grow(p_count)) {
			return false;
		}
	}
	return false;
}

void VirtualGeometryPool::Heap::free(uint32_t p_offset, uint32_t p_count) {
	uint32_t offset = p_offset;
	uint32_t size = p_count;

	// Coalesce with the following range.
	RBMap<uint32_t, uint32_t>::Element *next = free_ranges.find_closest(offset + size);
	if (next && next->key() == offset + size) {
		size += next->get();
		free_ranges.erase(next);
	}
	// Coalesce with the preceding range.
	RBMap<uint32_t, uint32_t>::Element *prev = free_ranges.find_closest(offset);
	if (prev && prev->key() + prev->get() == offset) {
		offset = prev->key();
		size += prev->get();
		free_ranges.erase(prev);
	}
	free_ranges.insert(offset, size);
}

uint32_t VirtualGeometryPool::Heap::get_used() const {
	uint32_t free_elements = 0;
	for (const KeyValue<uint32_t, uint32_t> &E : free_ranges) {
		free_elements += E.value;
	}
	return capacity - free_elements;
}

void VirtualGeometryPool::Heap::finalize() {
	if (buffer.is_valid()) {
		RD::get_singleton()->free_rid(buffer);
		buffer = RID();
	}
	capacity = 0;
	free_ranges.clear();
}

/* Pool */

void VirtualGeometryPool::queue_upload(Allocation *p_allocation, const Vector<uint8_t> &p_data) {
	ERR_FAIL_NULL(p_allocation);
	const VirtualGeometryFormat::Header *header = VirtualGeometryFormat::get_header(p_data.ptr(), p_data.size());
	ERR_FAIL_NULL(header);

	MutexLock lock(mutex);
	p_allocation->cluster_count = header->cluster_count;
	p_allocation->data_count = header->data_count;
	p_allocation->vertex_count = header->vertex_count;
	p_allocation->lod0_triangle_count = header->lod0_triangle_count;
	p_allocation->root_triangle_count = header->root_triangle_count;
	p_allocation->lod_level_count = header->lod_level_count;
	p_allocation->resident = false;

	PendingUpload upload;
	upload.allocation = p_allocation;
	upload.data = p_data;
	pending.push_back(upload);
}

void VirtualGeometryPool::free(Allocation *p_allocation) {
	ERR_FAIL_NULL(p_allocation);
	MutexLock lock(mutex);

	for (uint32_t i = 0; i < pending.size(); i++) {
		if (pending[i].allocation == p_allocation) {
			pending.remove_at_unordered(i);
			break;
		}
	}

	if (p_allocation->resident) {
		// The GPU may still be reading the old contents for frames in flight, but any
		// future write to this range is ordered after those reads by the render graph.
		clusters.free(p_allocation->cluster_offset, p_allocation->cluster_count);
		data.free(p_allocation->data_offset, p_allocation->data_count);
	}
	*p_allocation = Allocation();
}

void VirtualGeometryPool::flush_uploads() {
	MutexLock lock(mutex);
	if (pending.is_empty()) {
		return;
	}

	const RID old_cluster_buffer = clusters.get_buffer();
	const RID old_data_buffer = data.get_buffer();

	for (PendingUpload &upload : pending) {
		Allocation *allocation = upload.allocation;
		const uint8_t *blob = upload.data.ptr();
		const VirtualGeometryFormat::Header *header = VirtualGeometryFormat::get_header(blob, upload.data.size());
		ERR_CONTINUE(!header);

		uint32_t cluster_offset = 0;
		uint32_t data_offset = 0;
		if (!clusters.allocate(header->cluster_count, cluster_offset)) {
			continue;
		}
		if (!data.allocate(header->data_count, data_offset)) {
			clusters.free(cluster_offset, header->cluster_count);
			continue;
		}

		RD::get_singleton()->buffer_update(clusters.get_buffer(), cluster_offset * sizeof(VirtualGeometryFormat::Cluster), header->cluster_count * sizeof(VirtualGeometryFormat::Cluster), VirtualGeometryFormat::get_clusters(blob));
		RD::get_singleton()->buffer_update(data.get_buffer(), data_offset * sizeof(uint32_t), header->data_count * sizeof(uint32_t), VirtualGeometryFormat::get_cluster_data(blob));

		allocation->cluster_offset = cluster_offset;
		allocation->data_offset = data_offset;
		allocation->resident = true;
	}
	pending.clear();

	if (old_cluster_buffer != clusters.get_buffer() || old_data_buffer != data.get_buffer()) {
		version++;
	}
}

Vector<uint8_t> VirtualGeometryPool::read(const Allocation *p_allocation) const {
	ERR_FAIL_NULL_V(p_allocation, Vector<uint8_t>());
	MutexLock lock(mutex);

	if (!p_allocation->resident) {
		for (const PendingUpload &upload : pending) {
			if (upload.allocation == p_allocation) {
				return upload.data;
			}
		}
		return Vector<uint8_t>();
	}

	const Vector<uint8_t> cluster_bytes = RD::get_singleton()->buffer_get_data(clusters.get_buffer(), p_allocation->cluster_offset * sizeof(VirtualGeometryFormat::Cluster), p_allocation->cluster_count * sizeof(VirtualGeometryFormat::Cluster));
	const Vector<uint8_t> data_bytes = RD::get_singleton()->buffer_get_data(data.get_buffer(), p_allocation->data_offset * sizeof(uint32_t), p_allocation->data_count * sizeof(uint32_t));
	ERR_FAIL_COND_V(cluster_bytes.size() != int64_t(p_allocation->cluster_count * sizeof(VirtualGeometryFormat::Cluster)), Vector<uint8_t>());
	ERR_FAIL_COND_V(data_bytes.size() != int64_t(p_allocation->data_count * sizeof(uint32_t)), Vector<uint8_t>());

	// Rebuild the header from the cluster data, the same way the builder does.
	VirtualGeometryFormat::Header header;
	header.cluster_count = p_allocation->cluster_count;
	header.data_count = p_allocation->data_count;
	header.vertex_count = p_allocation->vertex_count;
	const VirtualGeometryFormat::Cluster *cluster_ptr = reinterpret_cast<const VirtualGeometryFormat::Cluster *>(cluster_bytes.ptr());
	uint32_t max_level = 0;
	for (uint32_t i = 0; i < header.cluster_count; i++) {
		const VirtualGeometryFormat::Cluster &c = cluster_ptr[i];
		max_level = MAX(max_level, c.get_lod_level());
		if (c.get_lod_level() == 0) {
			header.lod0_triangle_count += c.get_triangle_count();
		}
		if (c.parent_error >= VirtualGeometryFormat::ROOT_ERROR) {
			header.root_triangle_count += c.get_triangle_count();
		}
	}
	header.lod_level_count = max_level + 1;

	Vector<uint8_t> result;
	result.resize(sizeof(header) + cluster_bytes.size() + data_bytes.size());
	uint8_t *w = result.ptrw();
	memcpy(w, &header, sizeof(header));
	memcpy(w + sizeof(header), cluster_bytes.ptr(), cluster_bytes.size());
	memcpy(w + sizeof(header) + cluster_bytes.size(), data_bytes.ptr(), data_bytes.size());
	return result;
}

uint64_t VirtualGeometryPool::get_memory_usage() const {
	return uint64_t(clusters.get_capacity()) * sizeof(VirtualGeometryFormat::Cluster) + uint64_t(data.get_capacity()) * sizeof(uint32_t);
}

void VirtualGeometryPool::finalize() {
	MutexLock lock(mutex);
	pending.clear();
	clusters.finalize();
	data.finalize();
}
