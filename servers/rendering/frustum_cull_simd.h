/**************************************************************************/
/*  frustum_cull_simd.h                                                   */
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

#include "core/math/plane.h"
#include "core/typedefs.h"

// SIMD box-versus-frustum test used by the scene culler, which runs it for every
// instance against the camera frustum and every directional shadow cascade.
//
// Planes are stored as a structure of arrays in groups of four, so one group is
// tested with a handful of SSE/NEON instructions and no data-dependent loads.
// The box is tested in center/extent form:
//     outside = dot(n, center) - dot(|n|, extent) - d >= 0
// which is exactly the classic "nearest corner" test, without the per-plane
// corner gathers. Unused plane slots have n = 0 and d = 1, so they never reject.

#if !defined(REAL_T_IS_DOUBLE)
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define FRUSTUM_CULL_SIMD_SSE2
#include <emmintrin.h>
#elif (defined(__ARM_NEON) && defined(__aarch64__)) || defined(_M_ARM64)
#define FRUSTUM_CULL_SIMD_NEON
#include <arm_neon.h>
#endif
#endif

struct FrustumCullSIMD {
	static constexpr uint32_t MAX_PLANES = 16;

	float normal_x[MAX_PLANES];
	float normal_y[MAX_PLANES];
	float normal_z[MAX_PLANES];
	float abs_normal_x[MAX_PLANES];
	float abs_normal_y[MAX_PLANES];
	float abs_normal_z[MAX_PLANES];
	float distance[MAX_PLANES];
	uint32_t group_count = 0; // Groups of 4 planes.
	bool valid = false; // False when there are more planes than supported; use the scalar path.

	void setup(const Plane *p_planes, uint32_t p_count) {
		valid = p_count <= MAX_PLANES;
		group_count = (p_count + 3) / 4;
		for (uint32_t i = 0; i < MAX_PLANES; i++) {
			if (i < p_count) {
				const Plane &p = p_planes[i];
				normal_x[i] = float(p.normal.x);
				normal_y[i] = float(p.normal.y);
				normal_z[i] = float(p.normal.z);
				distance[i] = float(p.d);
			} else {
				normal_x[i] = 0.0f;
				normal_y[i] = 0.0f;
				normal_z[i] = 0.0f;
				distance[i] = 1.0f;
			}
			abs_normal_x[i] = Math::abs(normal_x[i]);
			abs_normal_y[i] = Math::abs(normal_y[i]);
			abs_normal_z[i] = Math::abs(normal_z[i]);
		}
	}

#if defined(FRUSTUM_CULL_SIMD_SSE2) || defined(FRUSTUM_CULL_SIMD_NEON)
	static constexpr bool has_simd = true;
#else
	static constexpr bool has_simd = false;
#endif

	// p_bounds: min x, y, z followed by max x, y, z.
	_ALWAYS_INLINE_ bool box_in_frustum(const float *p_bounds) const {
		const float cx = (p_bounds[0] + p_bounds[3]) * 0.5f;
		const float cy = (p_bounds[1] + p_bounds[4]) * 0.5f;
		const float cz = (p_bounds[2] + p_bounds[5]) * 0.5f;
		const float ex = (p_bounds[3] - p_bounds[0]) * 0.5f;
		const float ey = (p_bounds[4] - p_bounds[1]) * 0.5f;
		const float ez = (p_bounds[5] - p_bounds[2]) * 0.5f;

#if defined(FRUSTUM_CULL_SIMD_SSE2)
		const __m128 vcx = _mm_set1_ps(cx);
		const __m128 vcy = _mm_set1_ps(cy);
		const __m128 vcz = _mm_set1_ps(cz);
		const __m128 vex = _mm_set1_ps(ex);
		const __m128 vey = _mm_set1_ps(ey);
		const __m128 vez = _mm_set1_ps(ez);
		const __m128 zero = _mm_setzero_ps();
		for (uint32_t g = 0; g < group_count; g++) {
			const uint32_t o = g * 4;
			const __m128 center_distance = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_loadu_ps(normal_x + o), vcx), _mm_mul_ps(_mm_loadu_ps(normal_y + o), vcy)), _mm_mul_ps(_mm_loadu_ps(normal_z + o), vcz));
			const __m128 projected_radius = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_loadu_ps(abs_normal_x + o), vex), _mm_mul_ps(_mm_loadu_ps(abs_normal_y + o), vey)), _mm_mul_ps(_mm_loadu_ps(abs_normal_z + o), vez));
			const __m128 nearest = _mm_sub_ps(_mm_sub_ps(center_distance, projected_radius), _mm_loadu_ps(distance + o));
			if (_mm_movemask_ps(_mm_cmpge_ps(nearest, zero))) {
				return false;
			}
		}
		return true;
#elif defined(FRUSTUM_CULL_SIMD_NEON)
		const float32x4_t vcx = vdupq_n_f32(cx);
		const float32x4_t vcy = vdupq_n_f32(cy);
		const float32x4_t vcz = vdupq_n_f32(cz);
		const float32x4_t vex = vdupq_n_f32(ex);
		const float32x4_t vey = vdupq_n_f32(ey);
		const float32x4_t vez = vdupq_n_f32(ez);
		for (uint32_t g = 0; g < group_count; g++) {
			const uint32_t o = g * 4;
			float32x4_t center_distance = vmulq_f32(vld1q_f32(normal_x + o), vcx);
			center_distance = vmlaq_f32(center_distance, vld1q_f32(normal_y + o), vcy);
			center_distance = vmlaq_f32(center_distance, vld1q_f32(normal_z + o), vcz);
			float32x4_t projected_radius = vmulq_f32(vld1q_f32(abs_normal_x + o), vex);
			projected_radius = vmlaq_f32(projected_radius, vld1q_f32(abs_normal_y + o), vey);
			projected_radius = vmlaq_f32(projected_radius, vld1q_f32(abs_normal_z + o), vez);
			const float32x4_t nearest = vsubq_f32(vsubq_f32(center_distance, projected_radius), vld1q_f32(distance + o));
			const uint32x4_t outside = vcgeq_f32(nearest, vdupq_n_f32(0.0f));
			if (vmaxvq_u32(outside)) {
				return false;
			}
		}
		return true;
#else
		for (uint32_t i = 0; i < group_count * 4; i++) {
			const float nearest = normal_x[i] * cx + normal_y[i] * cy + normal_z[i] * cz - (abs_normal_x[i] * ex + abs_normal_y[i] * ey + abs_normal_z[i] * ez) - distance[i];
			if (nearest >= 0.0f) {
				return false;
			}
		}
		return true;
#endif
	}
};
