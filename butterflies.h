#pragma once
#include <vector>
typedef std::vector<std::pair<std::vector<float>,std::vector<float>>> Twiddles;

struct Precomputed {
    Twiddles twiddles;
    std::vector<float> window;
};

Precomputed precompute_fft_factors(int sampleCount);
void do_fft(float* reals, float* imags, int sampleCount);
void do_real_fft_precomputed(float* reals, float* imags, const Precomputed& precomputed, int sampleCount);
