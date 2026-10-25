/*
 * wav.h
 *
 * Copyright (C) 2026 RTLSDR-Airband contributors
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <https://www.gnu.org/licenses/>.
 */

#ifndef _WAV_H
#define _WAV_H

#include <cstddef>  // size_t
#include <cstdint>  // uint32_t
#include <cstdio>   // FILE

// WAV files written by file outputs: a canonical 44-byte header followed by
// 16-bit little-endian PCM, channels interleaved.
const size_t WAV_HEADER_SIZE = 44;

// Fill out[WAV_HEADER_SIZE] with a header for data_bytes of PCM.
void wav_header(unsigned char* out, int sample_rate, int channels, uint32_t data_bytes);

// Convert count samples in [-1, 1] to 16-bit little endian, interleaving left and
// right when right is not NULL. Out of range values are clipped and NaN becomes 0.
// out must hold count * 2 bytes per channel. Returns the number of bytes written.
size_t wav_samples(const float* left, const float* right, size_t count, unsigned char* out);

// True if f starts with a header written by wav_header for this format, so audio
// can be appended to it. The data sizes are not checked; wav_finalize sets them.
bool wav_is_appendable(FILE* f, int sample_rate, int channels);

// Set the header's size fields from the length of the file. Leaves the position at
// the end of the file. Returns false on an I/O error or a file shorter than a header.
bool wav_finalize(FILE* f);

#endif /* _WAV_H */
