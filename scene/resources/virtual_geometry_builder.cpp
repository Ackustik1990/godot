/**************************************************************************/
/*  virtual_geometry_builder.cpp                                          */
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

#include "virtual_geometry_builder.h"

#include "core/templates/local_vector.h"
#include "core/variant/variant.h"
#include "servers/rendering/rendering_server_enums.h"
#include "servers/rendering/virtual_geometry_format.h"

VirtualGeometryBuilder::BuildFunc VirtualGeometryBuilder::build_func = nullptr;

Vector<uint8_t> VirtualGeometryBuilder::build_from_arrays(const Array &p_arrays, const Settings &p_settings) {
	ERR_FAIL_NULL_V_MSG(build_func, Vector<uint8_t>(), "Virtual geometry is not available: the meshoptimizer module is disabled.");
	ERR_FAIL_COND_V(p_arrays.size() != RSE::ARRAY_MAX, Vector<uint8_t>());

	const PackedVector3Array vertices = p_arrays[RSE::ARRAY_VERTEX];
	const uint32_t vertex_count = vertices.size();
	ERR_FAIL_COND_V_MSG(vertex_count < 3, Vector<uint8_t>(), "Virtual geometry requires at least one triangle.");

	LocalVector<float> positions;
	positions.resize(vertex_count * 3);
	for (uint32_t i = 0; i < vertex_count; i++) {
		const Vector3 &v = vertices[i];
		positions[i * 3 + 0] = float(v.x);
		positions[i * 3 + 1] = float(v.y);
		positions[i * 3 + 2] = float(v.z);
	}

	LocalVector<float> normals;
	const PackedVector3Array normal_array = p_arrays[RSE::ARRAY_NORMAL];
	if (uint32_t(normal_array.size()) == vertex_count) {
		normals.resize(vertex_count * 3);
		for (uint32_t i = 0; i < vertex_count; i++) {
			const Vector3 &n = normal_array[i];
			normals[i * 3 + 0] = float(n.x);
			normals[i * 3 + 1] = float(n.y);
			normals[i * 3 + 2] = float(n.z);
		}
	}

	LocalVector<uint32_t> indices;
	const PackedInt32Array index_array = p_arrays[RSE::ARRAY_INDEX];
	if (index_array.size()) {
		ERR_FAIL_COND_V_MSG(index_array.size() % 3 != 0, Vector<uint8_t>(), "Virtual geometry requires a triangle list.");
		indices.resize(index_array.size());
		for (int i = 0; i < index_array.size(); i++) {
			ERR_FAIL_UNSIGNED_INDEX_V(uint32_t(index_array[i]), vertex_count, Vector<uint8_t>());
			indices[i] = uint32_t(index_array[i]);
		}
	} else {
		ERR_FAIL_COND_V_MSG(vertex_count % 3 != 0, Vector<uint8_t>(), "Virtual geometry requires a triangle list.");
		indices.resize(vertex_count);
		for (uint32_t i = 0; i < vertex_count; i++) {
			indices[i] = i;
		}
	}

	return build_func(positions.ptr(), normals.is_empty() ? nullptr : normals.ptr(), vertex_count, indices.ptr(), indices.size(), p_settings);
}

String VirtualGeometryBuilder::get_summary(const Vector<uint8_t> &p_data) {
	const VirtualGeometryFormat::Header *header = VirtualGeometryFormat::get_header(p_data.ptr(), p_data.size());
	if (!header) {
		return "Invalid virtual geometry data.";
	}
	return vformat("Virtual geometry: %d clusters, %d LOD levels, %d triangles at full detail, %d triangles at the coarsest cut, %s.",
			header->cluster_count, header->lod_level_count, header->lod0_triangle_count, header->root_triangle_count, String::humanize_size(p_data.size()));
}
