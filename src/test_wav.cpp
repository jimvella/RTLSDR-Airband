/*
 * test_wav.cpp
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

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "test_base_class.h"
#include "wav.h"

using namespace std;

class WavTest : public TestBaseClass {
   protected:
    FILE* open_temp(const char* name) {
        path = temp_dir + "/" + name;
        FILE* f = fopen(path.c_str(), "w+b");
        EXPECT_NE(f, nullptr);
        return f;
    }

    static uint32_t le32(const unsigned char* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
    static uint16_t le16(const unsigned char* p) { return p[0] | (p[1] << 8); }

    vector<unsigned char> read_all(FILE* f) {
        fseek(f, 0, SEEK_END);
        vector<unsigned char> data(ftell(f));
        fseek(f, 0, SEEK_SET);
        EXPECT_EQ(fread(data.data(), 1, data.size(), f), data.size());
        return data;
    }

    string path;
};

TEST_F(WavTest, header_mono_8k) {
    unsigned char h[WAV_HEADER_SIZE];
    wav_header(h, 8000, 1, 16000);
    EXPECT_EQ(memcmp(h, "RIFF", 4), 0);
    EXPECT_EQ(le32(h + 4), 36u + 16000u);
    EXPECT_EQ(memcmp(h + 8, "WAVEfmt ", 8), 0);
    EXPECT_EQ(le32(h + 16), 16u);     // fmt chunk size
    EXPECT_EQ(le16(h + 20), 1u);      // PCM
    EXPECT_EQ(le16(h + 22), 1u);      // channels
    EXPECT_EQ(le32(h + 24), 8000u);   // sample rate
    EXPECT_EQ(le32(h + 28), 16000u);  // byte rate
    EXPECT_EQ(le16(h + 32), 2u);      // block align
    EXPECT_EQ(le16(h + 34), 16u);     // bits per sample
    EXPECT_EQ(memcmp(h + 36, "data", 4), 0);
    EXPECT_EQ(le32(h + 40), 16000u);
}

TEST_F(WavTest, header_stereo_16k) {
    unsigned char h[WAV_HEADER_SIZE];
    wav_header(h, 16000, 2, 0);
    EXPECT_EQ(le32(h + 4), 36u);
    EXPECT_EQ(le16(h + 22), 2u);
    EXPECT_EQ(le32(h + 24), 16000u);
    EXPECT_EQ(le32(h + 28), 64000u);
    EXPECT_EQ(le16(h + 32), 4u);
    EXPECT_EQ(le32(h + 40), 0u);
}

TEST_F(WavTest, samples_mono_scale_and_clip) {
    const float in[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -3.0f, NAN};
    const size_t n = sizeof(in) / sizeof(in[0]);
    unsigned char out[2 * n];
    ASSERT_EQ(wav_samples(in, NULL, n, out), 2 * n);
    const int16_t expected[] = {0, 32767, -32767, 16384, 32767, -32767, 0};
    for (size_t i = 0; i < n; i++) {
        EXPECT_EQ((int16_t)le16(out + 2 * i), expected[i]) << "sample " << i;
    }
}

TEST_F(WavTest, samples_stereo_interleave) {
    const float left[] = {1.0f, 0.0f};
    const float right[] = {-1.0f, 0.5f};
    unsigned char out[8];
    ASSERT_EQ(wav_samples(left, right, 2, out), 8u);
    EXPECT_EQ((int16_t)le16(out + 0), 32767);
    EXPECT_EQ((int16_t)le16(out + 2), -32767);
    EXPECT_EQ((int16_t)le16(out + 4), 0);
    EXPECT_EQ((int16_t)le16(out + 6), 16384);
}

TEST_F(WavTest, finalize_sets_sizes_from_length) {
    FILE* f = open_temp("finalize.wav");
    unsigned char h[WAV_HEADER_SIZE];
    wav_header(h, 8000, 1, 0);
    fwrite(h, 1, sizeof(h), f);
    const unsigned char pcm[600] = {};
    fwrite(pcm, 1, sizeof(pcm), f);

    EXPECT_TRUE(wav_finalize(f));
    EXPECT_EQ(ftell(f), (long)(WAV_HEADER_SIZE + sizeof(pcm)));
    const vector<unsigned char> data = read_all(f);
    EXPECT_EQ(le32(data.data() + 4), 36u + 600u);
    EXPECT_EQ(le32(data.data() + 40), 600u);
    fclose(f);
}

TEST_F(WavTest, finalize_rejects_short_file) {
    FILE* f = open_temp("short.wav");
    fwrite("RIFF", 1, 4, f);
    EXPECT_FALSE(wav_finalize(f));
    fclose(f);
}

TEST_F(WavTest, appendable_matching_format) {
    FILE* f = open_temp("append.wav");
    unsigned char h[WAV_HEADER_SIZE];
    wav_header(h, 8000, 1, 0);  // sizes not yet set, as after a crash
    fwrite(h, 1, sizeof(h), f);
    EXPECT_TRUE(wav_is_appendable(f, 8000, 1));
    EXPECT_FALSE(wav_is_appendable(f, 16000, 1));
    EXPECT_FALSE(wav_is_appendable(f, 8000, 2));
    fclose(f);
}

TEST_F(WavTest, appendable_rejects_other_files) {
    FILE* f = open_temp("other.wav");
    const unsigned char mp3[WAV_HEADER_SIZE] = {0xff, 0xfb};
    fwrite(mp3, 1, sizeof(mp3), f);
    EXPECT_FALSE(wav_is_appendable(f, 8000, 1));
    fclose(f);

    FILE* g = open_temp("tiny.wav");
    fwrite("RIFF", 1, 4, g);
    EXPECT_FALSE(wav_is_appendable(g, 8000, 1));
    fclose(g);
}
