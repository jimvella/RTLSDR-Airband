/*
 * wav.cpp
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

#include "wav.h"

#include <cmath>    // lrintf, isnan
#include <cstring>  // memcpy, memcmp

static const int BITS_PER_SAMPLE = 16;

static void put_le16(unsigned char* p, uint32_t v) {
    p[0] = v & 0xff;
    p[1] = (v >> 8) & 0xff;
}

static void put_le32(unsigned char* p, uint32_t v) {
    put_le16(p, v & 0xffff);
    put_le16(p + 2, v >> 16);
}

void wav_header(unsigned char* out, int sample_rate, int channels, uint32_t data_bytes) {
    const uint32_t block_align = channels * BITS_PER_SAMPLE / 8;
    memcpy(out, "RIFF", 4);
    put_le32(out + 4, 36 + data_bytes);
    memcpy(out + 8, "WAVEfmt ", 8);
    put_le32(out + 16, 16);  // fmt chunk size
    put_le16(out + 20, 1);   // PCM
    put_le16(out + 22, channels);
    put_le32(out + 24, sample_rate);
    put_le32(out + 28, sample_rate * block_align);
    put_le16(out + 32, block_align);
    put_le16(out + 34, BITS_PER_SAMPLE);
    memcpy(out + 36, "data", 4);
    put_le32(out + 40, data_bytes);
}

static void put_sample(unsigned char* p, float v) {
    if (std::isnan(v)) {
        v = 0.0f;
    } else if (v > 1.0f) {
        v = 1.0f;
    } else if (v < -1.0f) {
        v = -1.0f;
    }
    put_le16(p, (uint16_t)(int16_t)lrintf(v * 32767.0f));
}

size_t wav_samples(const float* left, const float* right, size_t count, unsigned char* out) {
    unsigned char* p = out;
    for (size_t i = 0; i < count; i++) {
        put_sample(p, left[i]);
        p += 2;
        if (right) {
            put_sample(p, right[i]);
            p += 2;
        }
    }
    return p - out;
}

bool wav_is_appendable(FILE* f, int sample_rate, int channels) {
    unsigned char existing[WAV_HEADER_SIZE];
    if (fseek(f, 0, SEEK_SET) != 0 || fread(existing, 1, WAV_HEADER_SIZE, f) != WAV_HEADER_SIZE) {
        return false;
    }
    unsigned char expected[WAV_HEADER_SIZE];
    wav_header(expected, sample_rate, channels, 0);
    // compare everything but the two size fields
    return memcmp(existing, expected, 4) == 0 && memcmp(existing + 8, expected + 8, 32) == 0;
}

bool wav_finalize(FILE* f) {
    if (fseek(f, 0, SEEK_END) != 0) {
        return false;
    }
    const long length = ftell(f);
    if (length < (long)WAV_HEADER_SIZE) {
        return false;
    }
    unsigned char sizes[4];
    put_le32(sizes, length - 8);
    if (fseek(f, 4, SEEK_SET) != 0 || fwrite(sizes, 1, 4, f) != 4) {
        return false;
    }
    put_le32(sizes, length - WAV_HEADER_SIZE);
    if (fseek(f, 40, SEEK_SET) != 0 || fwrite(sizes, 1, 4, f) != 4) {
        return false;
    }
    return fseek(f, 0, SEEK_END) == 0;
}
