/**************************************************************************/
/*  rendering_upscaler.h                                                  */
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

#include "core/io/resource.h"
#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/object/gdvirtual.gen.h"
#include "core/object/ref_counted.h"

// Everything a temporal (possibly neural network based) upscaler needs for one view.
// Textures are RenderingDevice RIDs owned by the renderer and only valid during
// RenderingUpscaler::_upscale().
class RenderingUpscaleParameters : public RefCounted {
	GDCLASS(RenderingUpscaleParameters, RefCounted);

	friend class RenderingUpscaler;

protected:
	static void _bind_methods();

public:
	RID color; // HDR scene color, internal resolution.
	RID depth; // Depth buffer (reverse Z), internal resolution.
	RID velocity; // Motion vectors, internal resolution. See the class documentation for the encoding.
	RID reactive; // Optional reactivity mask (transparent and particle-heavy pixels), internal resolution.
	RID exposure; // Optional 1x1 R32F auto exposure luminance, invalid when auto exposure is disabled.
	RID output; // Storage texture at target resolution that must receive the upscaled image.

	Vector2i render_size;
	Vector2i target_size;
	Vector2 jitter; // Sub-pixel jitter applied to the projection, in pixels of the internal resolution.
	float z_near = 0.05;
	float z_far = 4000.0;
	float fov_y = 1.0; // Vertical field of view in radians.
	float delta_time = 0.0;
	float sharpness = 0.0;
	bool reset = false; // History must be discarded (camera cut, resize, first frame).
	uint32_t view = 0;
	uint32_t view_count = 1;
	uint64_t frame = 0;

	Projection projection;
	Projection previous_projection;
	Transform3D camera_transform;
	Transform3D previous_camera_transform;
	Projection reprojection; // Current clip space (with depth correction) to previous clip space.

	RID get_color() const { return color; }
	RID get_depth() const { return depth; }
	RID get_velocity() const { return velocity; }
	RID get_reactive() const { return reactive; }
	RID get_exposure() const { return exposure; }
	RID get_output() const { return output; }
	Vector2i get_render_size() const { return render_size; }
	Vector2i get_target_size() const { return target_size; }
	Vector2 get_jitter() const { return jitter; }
	float get_z_near() const { return z_near; }
	float get_z_far() const { return z_far; }
	float get_fov_y() const { return fov_y; }
	float get_delta_time() const { return delta_time; }
	float get_sharpness() const { return sharpness; }
	bool get_reset() const { return reset; }
	uint32_t get_view() const { return view; }
	uint32_t get_view_count() const { return view_count; }
	uint64_t get_frame() const { return frame; }
	Projection get_projection() const { return projection; }
	Projection get_previous_projection() const { return previous_projection; }
	Transform3D get_camera_transform() const { return camera_transform; }
	Transform3D get_previous_camera_transform() const { return previous_camera_transform; }
	Projection get_reprojection() const { return reprojection; }
};

// Base class for custom 3D upscalers, used with Viewport.SCALING_3D_MODE_CUSTOM.
//
// This is the integration point for vendor and neural network upscalers (DLSS,
// XeSS, FSR 3+, or custom models): implement it in a GDExtension or a script and
// record the upscaling work with the RenderingDevice. Upscalers that need to call
// into native SDKs can request a native command buffer instead. It's a resource so
// it can be configured in the inspector and assigned to viewports.
class RenderingUpscaler : public Resource {
	GDCLASS(RenderingUpscaler, Resource);

protected:
	static void _bind_methods();

	GDVIRTUAL0RC(String, _get_upscaler_name)
	GDVIRTUAL0RC(bool, _is_supported)
	GDVIRTUAL2RC(int, _get_jitter_phase_count, Vector2i, Vector2i)
	GDVIRTUAL0RC(bool, _uses_native_commands)
	GDVIRTUAL1(_upscale, Ref<RenderingUpscaleParameters>)
	GDVIRTUAL2(_record_native_commands, int64_t, Ref<RenderingUpscaleParameters>)

public:
	virtual String get_upscaler_name() const;
	virtual bool is_supported() const;
	// Number of distinct jitter offsets before the sequence repeats.
	virtual uint32_t get_jitter_phase_count(const Size2i &p_render_size, const Size2i &p_target_size) const;
	virtual bool uses_native_commands() const;
	// Records the work with the RenderingDevice. Called on the render thread.
	virtual void upscale(const Ref<RenderingUpscaleParameters> &p_parameters);
	// Called while the frame's command buffer is being recorded, with the native
	// handle of that command buffer (VkCommandBuffer, ID3D12GraphicsCommandList or
	// MTLCommandBuffer). Only used when uses_native_commands() returns true.
	virtual void record_native_commands(uint64_t p_command_buffer, const Ref<RenderingUpscaleParameters> &p_parameters);

	static uint32_t get_default_jitter_phase_count(const Size2i &p_render_size, const Size2i &p_target_size);
};
