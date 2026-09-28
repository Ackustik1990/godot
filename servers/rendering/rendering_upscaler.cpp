/**************************************************************************/
/*  rendering_upscaler.cpp                                                */
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

#include "rendering_upscaler.h"

#include "core/object/class_db.h"

#include <cmath>

void RenderingUpscaleParameters::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_color"), &RenderingUpscaleParameters::get_color);
	ClassDB::bind_method(D_METHOD("get_depth"), &RenderingUpscaleParameters::get_depth);
	ClassDB::bind_method(D_METHOD("get_velocity"), &RenderingUpscaleParameters::get_velocity);
	ClassDB::bind_method(D_METHOD("get_reactive"), &RenderingUpscaleParameters::get_reactive);
	ClassDB::bind_method(D_METHOD("get_exposure"), &RenderingUpscaleParameters::get_exposure);
	ClassDB::bind_method(D_METHOD("get_output"), &RenderingUpscaleParameters::get_output);
	ClassDB::bind_method(D_METHOD("get_render_size"), &RenderingUpscaleParameters::get_render_size);
	ClassDB::bind_method(D_METHOD("get_target_size"), &RenderingUpscaleParameters::get_target_size);
	ClassDB::bind_method(D_METHOD("get_jitter"), &RenderingUpscaleParameters::get_jitter);
	ClassDB::bind_method(D_METHOD("get_z_near"), &RenderingUpscaleParameters::get_z_near);
	ClassDB::bind_method(D_METHOD("get_z_far"), &RenderingUpscaleParameters::get_z_far);
	ClassDB::bind_method(D_METHOD("get_fov_y"), &RenderingUpscaleParameters::get_fov_y);
	ClassDB::bind_method(D_METHOD("get_delta_time"), &RenderingUpscaleParameters::get_delta_time);
	ClassDB::bind_method(D_METHOD("get_sharpness"), &RenderingUpscaleParameters::get_sharpness);
	ClassDB::bind_method(D_METHOD("get_reset"), &RenderingUpscaleParameters::get_reset);
	ClassDB::bind_method(D_METHOD("get_view"), &RenderingUpscaleParameters::get_view);
	ClassDB::bind_method(D_METHOD("get_view_count"), &RenderingUpscaleParameters::get_view_count);
	ClassDB::bind_method(D_METHOD("get_frame"), &RenderingUpscaleParameters::get_frame);
	ClassDB::bind_method(D_METHOD("get_projection"), &RenderingUpscaleParameters::get_projection);
	ClassDB::bind_method(D_METHOD("get_previous_projection"), &RenderingUpscaleParameters::get_previous_projection);
	ClassDB::bind_method(D_METHOD("get_camera_transform"), &RenderingUpscaleParameters::get_camera_transform);
	ClassDB::bind_method(D_METHOD("get_previous_camera_transform"), &RenderingUpscaleParameters::get_previous_camera_transform);
	ClassDB::bind_method(D_METHOD("get_reprojection"), &RenderingUpscaleParameters::get_reprojection);
}

uint32_t RenderingUpscaler::get_default_jitter_phase_count(const Size2i &p_render_size, const Size2i &p_target_size) {
	// Same as ffxFsr2GetJitterPhaseCount(), which suits most temporal upscalers.
	if (p_render_size.width <= 0) {
		return 8;
	}
	return uint32_t(8.0f * std::pow(float(p_target_size.width) / float(p_render_size.width), 2.0f));
}

String RenderingUpscaler::get_upscaler_name() const {
	String name;
	if (GDVIRTUAL_CALL(_get_upscaler_name, name)) {
		return name;
	}
	return get_class();
}

bool RenderingUpscaler::is_supported() const {
	bool supported = false;
	if (GDVIRTUAL_CALL(_is_supported, supported)) {
		return supported;
	}
	// Without an implementation of _upscale() there's nothing to run.
	return GDVIRTUAL_IS_OVERRIDDEN(_upscale) || GDVIRTUAL_IS_OVERRIDDEN(_record_native_commands);
}

uint32_t RenderingUpscaler::get_jitter_phase_count(const Size2i &p_render_size, const Size2i &p_target_size) const {
	int count = 0;
	if (GDVIRTUAL_CALL(_get_jitter_phase_count, Vector2i(p_render_size), Vector2i(p_target_size), count) && count > 0) {
		return count;
	}
	return get_default_jitter_phase_count(p_render_size, p_target_size);
}

bool RenderingUpscaler::uses_native_commands() const {
	bool native = false;
	GDVIRTUAL_CALL(_uses_native_commands, native);
	return native;
}

void RenderingUpscaler::upscale(const Ref<RenderingUpscaleParameters> &p_parameters) {
	GDVIRTUAL_CALL(_upscale, p_parameters);
}

void RenderingUpscaler::record_native_commands(uint64_t p_command_buffer, const Ref<RenderingUpscaleParameters> &p_parameters) {
	GDVIRTUAL_CALL(_record_native_commands, int64_t(p_command_buffer), p_parameters);
}

void RenderingUpscaler::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_upscaler_name"), &RenderingUpscaler::get_upscaler_name);
	ClassDB::bind_method(D_METHOD("is_supported"), &RenderingUpscaler::is_supported);

	GDVIRTUAL_BIND(_get_upscaler_name);
	GDVIRTUAL_BIND(_is_supported);
	GDVIRTUAL_BIND(_get_jitter_phase_count, "render_size", "target_size");
	GDVIRTUAL_BIND(_uses_native_commands);
	GDVIRTUAL_BIND(_upscale, "parameters");
	GDVIRTUAL_BIND(_record_native_commands, "command_buffer", "parameters");
}
