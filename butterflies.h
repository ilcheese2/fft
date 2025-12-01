#pragma once

typedef std::vector<std::pair<std::vector<float>,std::vector<float>>> Twiddles;
Twiddles generate_twiddles(int sampleCount);
void do_fft(float* reals, float* imags, int sampleCount);
