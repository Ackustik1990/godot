/**************************************************************************/
/*  virtual_geometry_format.h                                             */
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

#include <cstdint>
#include <cstring>

// Binary layout of the "virtual geometry" data attached to a mesh surface.
//
// Virtual geometry splits a surface into small clusters (meshlets) and builds a
// hierarchy of simplified cluster groups on top of them (a cluster DAG, similar
// in spirit to Nanite). At render time a compute pass selects, per instance and
// per view, the set of clusters whose simplification error projects below a
// pixel threshold, culls them against the frustum and their normal cone, and
// compacts their triangles into an index buffer that is drawn indirectly with
// the regular material pipelines.
//
// The simplifier never creates new vertices, so every cluster of every LOD
// level indexes the original vertex buffer of the surface. This keeps the
// feature compatible with every material, vertex format and render pass.
//
// Layout of the blob:
//   Header
//   Cluster[cluster_count]
//   uint32_t data[data_count]  (per-cluster vertex references and packed triangles)
//
// All values are little-endian. The cluster struct matches the std430 layout
// used by the culling shader (virtual_geometry.glsl).

namespace VirtualGeometryFormat {

static constexpr uint32_t MAGIC = 0x4f454756; // "VGEO"
static constexpr uint32_t VERSION = 1;

static constexpr uint32_t MAX_CLUSTER_VERTICES = 128;
static constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128;

// Parent error of root clusters (clusters that are never replaced by a coarser version).
static constexpr float ROOT_ERROR = 3.0e38f;

struct Header {
	uint32_t magic = MAGIC;
	uint32_t version = VERSION;
	uint32_t cluster_count = 0;
	uint32_t data_count = 0; // In 32-bit words.
	uint32_t vertex_count = 0; // Vertex count of the surface this data was built for.
	uint32_t lod_level_count = 0;
	uint32_t lod0_triangle_count = 0; // Upper bound of the triangles selected by any LOD cut.
	uint32_t root_triangle_count = 0; // Triangles in the coarsest possible cut.
};

static_assert(sizeof(Header) == 32);

enum ClusterPacking {
	CLUSTER_TRIANGLE_COUNT_SHIFT = 0,
	CLUSTER_VERTEX_COUNT_SHIFT = 8,
	CLUSTER_LOD_LEVEL_SHIFT = 16,
	CLUSTER_COUNT_MASK = 0xFF,
};

struct Cluster {
	float bounds[4]; // Culling sphere: center (xyz) and radius (w), in mesh space.
	float lod_sphere[4]; // Sphere of the group this cluster was generated from.
	float parent_sphere[4]; // Sphere of the group this cluster was simplified into.
	float lod_error; // Absolute simplification error of this cluster, in mesh units.
	float parent_error; // Error of the parent group (ROOT_ERROR if this is a root).
	uint32_t cone; // Normal cone for backface culling: axis (xyz) and cutoff (w) as snorm8x4. A cutoff of 127 disables it.
	uint32_t counts; // Triangle count, vertex count and LOD level, see ClusterPacking. Counts are stored minus one.
	uint32_t vertex_offset; // Offset (in words, relative to the data array) of the vertex references.
	uint32_t triangle_offset; // Offset (in words, relative to the data array) of the packed triangles (3 x 8 bits each).
	uint32_t reserved[2];

	inline uint32_t get_triangle_count() const { return ((counts >> CLUSTER_TRIANGLE_COUNT_SHIFT) & CLUSTER_COUNT_MASK) + 1; }
	inline uint32_t get_vertex_count() const { return ((counts >> CLUSTER_VERTEX_COUNT_SHIFT) & CLUSTER_COUNT_MASK) + 1; }
	inline uint32_t get_lod_level() const { return (counts >> CLUSTER_LOD_LEVEL_SHIFT) & CLUSTER_COUNT_MASK; }
};

static_assert(sizeof(Cluster) == 80);

inline const Header *get_header(const uint8_t *p_data, uint64_t p_size) {
	if (p_data == nullptr || p_size < sizeof(Header)) {
		return nullptr;
	}
	const Header *header = reinterpret_cast<const Header *>(p_data);
	if (header->magic != MAGIC || header->version != VERSION) {
		return nullptr;
	}
	const uint64_t expected = sizeof(Header) + uint64_t(header->cluster_count) * sizeof(Cluster) + uint64_t(header->data_count) * sizeof(uint32_t);
	if (p_size != expected || header->cluster_count == 0) {
		return nullptr;
	}
	return header;
}

inline const Cluster *get_clusters(const uint8_t *p_data) {
	return reinterpret_cast<const Cluster *>(p_data + sizeof(Header));
}

inline const uint32_t *get_cluster_data(const uint8_t *p_data) {
	const Header *header = reinterpret_cast<const Header *>(p_data);
	return reinterpret_cast<const uint32_t *>(p_data + sizeof(Header) + header->cluster_count * sizeof(Cluster));
}

// Full validation, including every cluster range and every vertex reference.
// Use before uploading data that comes from untrusted sources (e.g. resource files).
inline bool validate(const uint8_t *p_data, uint64_t p_size, uint32_t p_vertex_count) {
	const Header *header = get_header(p_data, p_size);
	if (!header || header->vertex_count != p_vertex_count) {
		return false;
	}
	const Cluster *clusters = get_clusters(p_data);
	const uint32_t *data = get_cluster_data(p_data);
	for (uint32_t i = 0; i < header->cluster_count; i++) {
		const Cluster &c = clusters[i];
		const uint32_t vertex_count = c.get_vertex_count();
		const uint32_t triangle_count = c.get_triangle_count();
		if (vertex_count > MAX_CLUSTER_VERTICES || triangle_count > MAX_CLUSTER_TRIANGLES) {
			return false;
		}
		if (uint64_t(c.vertex_offset) + vertex_count > header->data_count || uint64_t(c.triangle_offset) + triangle_count > header->data_count) {
			return false;
		}
		for (uint32_t v = 0; v < vertex_count; v++) {
			if (data[c.vertex_offset + v] >= p_vertex_count) {
				return false;
			}
		}
		for (uint32_t t = 0; t < triangle_count; t++) {
			const uint32_t tri = data[c.triangle_offset + t];
			if ((tri & 0xFF) >= vertex_count || ((tri >> 8) & 0xFF) >= vertex_count || ((tri >> 16) & 0xFF) >= vertex_count) {
				return false;
			}
		}
		if (!(c.bounds[3] >= 0.0f) || !(c.lod_error >= 0.0f) || !(c.parent_error >= c.lod_error)) {
			return false;
		}
	}
	return true;
}

} // namespace VirtualGeometryFormat
