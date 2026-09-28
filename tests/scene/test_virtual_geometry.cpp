/**************************************************************************/
/*  test_virtual_geometry.cpp                                             */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_virtual_geometry)

#ifndef _3D_DISABLED

#include "core/math/random_pcg.h"
#include "core/templates/hash_map.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/mesh.h"
#include "scene/resources/virtual_geometry_builder.h"
#include "servers/rendering/virtual_geometry_format.h"

namespace TestVirtualGeometry {

using VirtualGeometryFormat::Cluster;
using VirtualGeometryFormat::Header;

static Array make_sphere_arrays(int p_radial_segments, int p_rings) {
	Ref<SphereMesh> sphere;
	sphere.instantiate();
	sphere->set_radius(1.0);
	sphere->set_height(2.0);
	sphere->set_radial_segments(p_radial_segments);
	sphere->set_rings(p_rings);
	return sphere->get_mesh_arrays();
}

// CPU replica of is_fine_enough() in virtual_geometry.glsl, for an identity transform.
static bool is_fine_enough(const float *p_sphere, float p_error, const Vector3 &p_camera, float p_lod_factor) {
	if (p_error <= 0.0f) {
		return true;
	}
	if (p_error >= VirtualGeometryFormat::ROOT_ERROR) {
		return false;
	}
	const float distance = p_camera.distance_to(Vector3(p_sphere[0], p_sphere[1], p_sphere[2])) - p_sphere[3];
	return p_error * p_lod_factor <= distance;
}

static void select_cut(const Vector<uint8_t> &p_data, const Vector3 &p_camera, float p_lod_factor, LocalVector<uint32_t> &r_indices) {
	const Header *header = VirtualGeometryFormat::get_header(p_data.ptr(), p_data.size());
	const Cluster *clusters = VirtualGeometryFormat::get_clusters(p_data.ptr());
	const uint32_t *data = VirtualGeometryFormat::get_cluster_data(p_data.ptr());
	r_indices.clear();
	for (uint32_t i = 0; i < header->cluster_count; i++) {
		const Cluster &c = clusters[i];
		if (!is_fine_enough(c.lod_sphere, c.lod_error, p_camera, p_lod_factor) || is_fine_enough(c.parent_sphere, c.parent_error, p_camera, p_lod_factor)) {
			continue;
		}
		for (uint32_t t = 0; t < c.get_triangle_count(); t++) {
			const uint32_t tri = data[c.triangle_offset + t];
			r_indices.push_back(data[c.vertex_offset + (tri & 0xFF)]);
			r_indices.push_back(data[c.vertex_offset + ((tri >> 8) & 0xFF)]);
			r_indices.push_back(data[c.vertex_offset + ((tri >> 16) & 0xFF)]);
		}
	}
}

// Returns the number of edges that aren't shared by exactly two consistently oriented
// triangles. Vertices are welded by position, so UV seams don't count as borders.
static int count_open_edges(const PackedVector3Array &p_vertices, const LocalVector<uint32_t> &p_indices) {
	HashMap<Vector3, uint32_t> position_ids;
	LocalVector<uint32_t> remap;
	remap.resize(p_vertices.size());
	for (int i = 0; i < p_vertices.size(); i++) {
		HashMap<Vector3, uint32_t>::Iterator E = position_ids.find(p_vertices[i]);
		if (E) {
			remap[i] = E->value;
		} else {
			const uint32_t id = position_ids.size();
			position_ids.insert(p_vertices[i], id);
			remap[i] = id;
		}
	}

	HashMap<uint64_t, int> directed_edges;
	for (uint32_t i = 0; i < p_indices.size(); i += 3) {
		const uint32_t v[3] = { remap[p_indices[i]], remap[p_indices[i + 1]], remap[p_indices[i + 2]] };
		if (v[0] == v[1] || v[1] == v[2] || v[0] == v[2]) {
			continue; // Degenerate (e.g. at the poles).
		}
		for (int e = 0; e < 3; e++) {
			const uint64_t key = (uint64_t(v[e]) << 32) | v[(e + 1) % 3];
			directed_edges[key]++;
		}
	}

	int open_edges = 0;
	for (const KeyValue<uint64_t, int> &E : directed_edges) {
		const uint64_t reverse = (E.key >> 32) | (E.key << 32);
		HashMap<uint64_t, int>::ConstIterator R = directed_edges.find(reverse);
		if (E.value != 1 || !R || R->value != 1) {
			open_edges++;
		}
	}
	return open_edges;
}

TEST_CASE("[VirtualGeometry] Cluster hierarchy is valid and monotonic") {
	REQUIRE_MESSAGE(VirtualGeometryBuilder::is_available(), "The meshoptimizer module is required for virtual geometry.");

	const Array arrays = make_sphere_arrays(128, 64);
	const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	const PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];

	const Vector<uint8_t> data = VirtualGeometryBuilder::build_from_arrays(arrays);
	REQUIRE(!data.is_empty());
	CHECK(VirtualGeometryFormat::validate(data.ptr(), data.size(), vertices.size()));

	const Header *header = VirtualGeometryFormat::get_header(data.ptr(), data.size());
	REQUIRE(header != nullptr);
	CHECK(header->lod_level_count > 3);
	CHECK(header->lod0_triangle_count <= uint32_t(indices.size() / 3));
	CHECK(header->lod0_triangle_count > uint32_t(indices.size() / 3) * 9 / 10);
	CHECK(header->root_triangle_count < header->lod0_triangle_count / 8);

	const Cluster *clusters = VirtualGeometryFormat::get_clusters(data.ptr());
	for (uint32_t i = 0; i < header->cluster_count; i++) {
		const Cluster &c = clusters[i];
		CHECK(c.parent_error >= c.lod_error);
		if (c.parent_error < VirtualGeometryFormat::ROOT_ERROR) {
			// The parent sphere must contain the cluster's own LOD sphere.
			const float distance = Vector3(c.parent_sphere[0], c.parent_sphere[1], c.parent_sphere[2]).distance_to(Vector3(c.lod_sphere[0], c.lod_sphere[1], c.lod_sphere[2]));
			CHECK(distance + c.lod_sphere[3] <= c.parent_sphere[3] * 1.0001f + 1e-6f);
		}
		if (c.get_lod_level() == 0) {
			CHECK(c.lod_error == 0.0f);
		}
	}
}

TEST_CASE("[VirtualGeometry] Every LOD cut is watertight") {
	REQUIRE(VirtualGeometryBuilder::is_available());

	const Array arrays = make_sphere_arrays(128, 64);
	const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	const Vector<uint8_t> data = VirtualGeometryBuilder::build_from_arrays(arrays);
	REQUIRE(!data.is_empty());
	const Header *header = VirtualGeometryFormat::get_header(data.ptr(), data.size());

	LocalVector<uint32_t> cut;
	uint32_t previous_triangles = UINT32_MAX;
	// From full detail to the coarsest cut, with the camera close to the surface so
	// that different parts of the mesh use different levels of detail.
	const float lod_factors[] = { 1e9f, 5000.0f, 1000.0f, 200.0f, 50.0f, 10.0f, 1.0f, 0.0001f };
	for (float lod_factor : lod_factors) {
		select_cut(data, Vector3(0.0, 0.0, 1.3), lod_factor, cut);
		const uint32_t triangles = cut.size() / 3;
		CHECK_MESSAGE(count_open_edges(vertices, cut) == 0, vformat("Cracks found in the LOD cut for factor %f.", lod_factor));
		CHECK(triangles > 0);
		CHECK(triangles <= previous_triangles);
		previous_triangles = triangles;
	}

	select_cut(data, Vector3(0.0, 0.0, 1.3), 1e9f, cut);
	CHECK(cut.size() / 3 == header->lod0_triangle_count);
	select_cut(data, Vector3(0.0, 0.0, 1000.0), 1.0f, cut);
	CHECK(cut.size() / 3 == header->root_triangle_count);
}

TEST_CASE("[VirtualGeometry] Normal cones only reject back-facing clusters") {
	REQUIRE(VirtualGeometryBuilder::is_available());

	const Array arrays = make_sphere_arrays(64, 32);
	const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	const Vector<uint8_t> data = VirtualGeometryBuilder::build_from_arrays(arrays);
	REQUIRE(!data.is_empty());

	const Header *header = VirtualGeometryFormat::get_header(data.ptr(), data.size());
	const Cluster *clusters = VirtualGeometryFormat::get_clusters(data.ptr());
	const uint32_t *cluster_data = VirtualGeometryFormat::get_cluster_data(data.ptr());

	RandomPCG rng(1234);
	int culled = 0;
	int wrongly_culled = 0;
	int front_facing_triangles = 0;
	int triangles_checked = 0;

	for (int camera_index = 0; camera_index < 32; camera_index++) {
		const Vector3 camera = Vector3(rng.randf() * 2.0 - 1.0, rng.randf() * 2.0 - 1.0, rng.randf() * 2.0 - 1.0).normalized() * (1.5 + rng.randf() * 4.0);

		for (uint32_t i = 0; i < header->cluster_count; i++) {
			const Cluster &c = clusters[i];
			const int8_t cutoff_s8 = int8_t(c.cone >> 24);
			if (cutoff_s8 >= 127) {
				continue;
			}
			const Vector3 axis = Vector3(int8_t(c.cone & 0xFF), int8_t((c.cone >> 8) & 0xFF), int8_t((c.cone >> 16) & 0xFF)) / 127.0;
			const float cutoff = cutoff_s8 / 127.0f;
			const Vector3 center(c.bounds[0], c.bounds[1], c.bounds[2]);
			const Vector3 to_cluster = center - camera;
			const bool cone_culled = to_cluster.dot(axis) >= cutoff * to_cluster.length() + c.bounds[3];

			for (uint32_t t = 0; t < c.get_triangle_count(); t++) {
				const uint32_t tri = cluster_data[c.triangle_offset + t];
				const Vector3 a = vertices[cluster_data[c.vertex_offset + (tri & 0xFF)]];
				const Vector3 b = vertices[cluster_data[c.vertex_offset + ((tri >> 8) & 0xFF)]];
				const Vector3 d = vertices[cluster_data[c.vertex_offset + ((tri >> 16) & 0xFF)]];
				// Godot uses clockwise winding for front faces.
				const Vector3 normal = (d - a).cross(b - a);
				if (normal.length_squared() < 1e-12) {
					continue;
				}
				triangles_checked++;
				if (normal.dot((a + b + d) / 3.0) > 0.0) {
					front_facing_triangles++; // Faces outwards, as expected for a sphere.
				}
				if (cone_culled && normal.dot(camera - a) > 1e-6) {
					wrongly_culled++;
				}
			}
			if (cone_culled) {
				culled++;
			}
		}
	}

	// Sanity check of the winding convention used by the test itself.
	CHECK(front_facing_triangles == triangles_checked);
	CHECK(culled > 0);
	CHECK(wrongly_culled == 0);
}

TEST_CASE("[VirtualGeometry] Validation rejects corrupted data") {
	REQUIRE(VirtualGeometryBuilder::is_available());

	const Array arrays = make_sphere_arrays(32, 16);
	const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	Vector<uint8_t> data = VirtualGeometryBuilder::build_from_arrays(arrays);
	REQUIRE(!data.is_empty());
	CHECK(VirtualGeometryFormat::validate(data.ptr(), data.size(), vertices.size()));
	CHECK_FALSE(VirtualGeometryFormat::validate(data.ptr(), data.size(), vertices.size() - 1));
	CHECK_FALSE(VirtualGeometryFormat::validate(data.ptr(), data.size() - 4, vertices.size()));

	const Header *header = VirtualGeometryFormat::get_header(data.ptr(), data.size());
	Cluster *clusters = reinterpret_cast<Cluster *>(data.ptrw() + sizeof(Header));
	uint32_t *cluster_data = reinterpret_cast<uint32_t *>(data.ptrw() + sizeof(Header) + header->cluster_count * sizeof(Cluster));
	const uint32_t saved = cluster_data[clusters[0].vertex_offset];
	cluster_data[clusters[0].vertex_offset] = vertices.size();
	CHECK_FALSE(VirtualGeometryFormat::validate(data.ptr(), data.size(), vertices.size()));
	cluster_data[clusters[0].vertex_offset] = saved;
	clusters[0].triangle_offset = header->data_count;
	CHECK_FALSE(VirtualGeometryFormat::validate(data.ptr(), data.size(), vertices.size()));
}

TEST_CASE("[SceneTree][VirtualGeometry] ArrayMesh generates and preserves virtual geometry") {
	REQUIRE(VirtualGeometryBuilder::is_available());

	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, make_sphere_arrays(32, 16));
	CHECK_FALSE(mesh->surface_has_virtual_geometry(0));

	CHECK(mesh->generate_virtual_geometry() == OK);
	CHECK(mesh->surface_has_virtual_geometry(0));

	const Array surfaces = mesh->call("_get_surfaces");
	REQUIRE(surfaces.size() == 1);
	const Dictionary surface = surfaces[0];
	REQUIRE(surface.has("virtual_geometry"));
	const Vector<uint8_t> blob = surface["virtual_geometry"];
	CHECK(VirtualGeometryFormat::validate(blob.ptr(), blob.size(), int(surface["vertex_count"])));

	// Round trip through the serialized form.
	Ref<ArrayMesh> copy;
	copy.instantiate();
	copy->call("_set_surfaces", surfaces);
	CHECK(copy->surface_has_virtual_geometry(0));

	mesh->clear_virtual_geometry();
	CHECK_FALSE(mesh->surface_has_virtual_geometry(0));
}

} // namespace TestVirtualGeometry

#endif // _3D_DISABLED
