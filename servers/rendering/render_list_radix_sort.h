/**************************************************************************/
/*  render_list_radix_sort.h                                              */
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

#include "core/templates/local_vector.h"

// Stable LSD radix sort for render lists sorted by a 128-bit key (key2 is the most
// significant word). Keys are copied next to the element pointers once, so the
// passes stream through memory instead of chasing pointers like a comparison sort
// does, and 8-bit digits that are identical for every element are skipped.
//
// Measured about 3x faster than introsort for 10,000+ elements. For small lists a
// comparison sort is faster, so callers should keep using it below
// RENDER_LIST_RADIX_SORT_THRESHOLD elements.

static constexpr uint32_t RENDER_LIST_RADIX_SORT_THRESHOLD = 1024;

class RenderListRadixSorter {
	struct Item {
		uint64_t key1;
		uint64_t key2;
		void *element;
	};

	LocalVector<Item> items;
	LocalVector<Item> scratch;

	template <int WORD>
	static _FORCE_INLINE_ uint64_t _key(const Item &p_item) {
		if constexpr (WORD == 0) {
			return p_item.key1;
		} else {
			return p_item.key2;
		}
	}

	template <int WORD>
	static void _sort_word(Item *&r_src, Item *&r_dst, uint32_t p_count, uint64_t p_varying_bits) {
		for (uint32_t shift = 0; shift < 64; shift += 8) {
			if (((p_varying_bits >> shift) & 0xFF) == 0) {
				continue; // Same digit for every element.
			}
			uint32_t offsets[256] = {};
			for (uint32_t i = 0; i < p_count; i++) {
				offsets[(_key<WORD>(r_src[i]) >> shift) & 0xFF]++;
			}
			uint32_t sum = 0;
			for (uint32_t d = 0; d < 256; d++) {
				const uint32_t c = offsets[d];
				offsets[d] = sum;
				sum += c;
			}
			for (uint32_t i = 0; i < p_count; i++) {
				r_dst[offsets[(_key<WORD>(r_src[i]) >> shift) & 0xFF]++] = r_src[i];
			}
			SWAP(r_src, r_dst);
		}
	}

public:
	// T must expose `sort.sort_key1` and `sort.sort_key2`.
	template <typename T>
	void sort(T **p_elements, uint32_t p_count) {
		if (p_count < 2) {
			return;
		}
		items.resize(p_count);
		scratch.resize(p_count);

		uint64_t and1 = ~uint64_t(0);
		uint64_t or1 = 0;
		uint64_t and2 = ~uint64_t(0);
		uint64_t or2 = 0;
		for (uint32_t i = 0; i < p_count; i++) {
			Item &item = items[i];
			item.key1 = p_elements[i]->sort.sort_key1;
			item.key2 = p_elements[i]->sort.sort_key2;
			item.element = p_elements[i];
			and1 &= item.key1;
			or1 |= item.key1;
			and2 &= item.key2;
			or2 |= item.key2;
		}

		Item *src = items.ptr();
		Item *dst = scratch.ptr();
		// Least significant word first; each pass is stable.
		_sort_word<0>(src, dst, p_count, and1 ^ or1);
		_sort_word<1>(src, dst, p_count, and2 ^ or2);

		for (uint32_t i = 0; i < p_count; i++) {
			p_elements[i] = static_cast<T *>(src[i].element);
		}
	}
};
