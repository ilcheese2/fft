#include <cassert>
#include <iostream>
#include <complex>
#include "fftw3.h"
#include <chrono>
#include "butterflies.h"
#include <map>

typedef struct wav_header { // https://gist.github.com/Jon-Schneider/8b7c53d27a7a13346a643dac9c19d34f
    // RIFF Header
    char riff_header[4]; // Contains "RIFF"
    int wav_size; // Size of the wav portion of the file, which follows the first 8 bytes. File size - 8
    char wave_header[4]; // Contains "WAVE"

    // Format Header
    char fmt_header[4]; // Contains "fmt " (includes trailing space)
    int fmt_chunk_size; // Should be 16 for PCM
    short audio_format; // Should be 1 for PCM. 3 for IEEE Float
    short num_channels;
    int sample_rate;
    int byte_rate; // Number of bytes per second. sample_rate * num_channels * Bytes Per Sample
    short sample_alignment; // num_channels * Bytes Per Sample
    short bit_depth; // Number of bits per sample
} wav_header;

std::vector<std::complex<float>> dft(const std::vector<std::complex<float>>& samples, int sampleCount) {
    using namespace std::complex_literals;
    const std::complex<float> W = exp(-2.0if * (float) M_PI  / (float) sampleCount);
    auto values = std::vector<std::complex<float>>(sampleCount);
    for (int i = 0; i < sampleCount; i++) {
        for (int j = 0; j < sampleCount; j++) {
            values[i] += samples[j] * pow(W, (float)i*j);
        }
    }
    return values;
}

std::vector<std::complex<float>> W_map = {};


std::vector<std::complex<float>> recursive_fft(std::vector<std::complex<float>> samples, int sampleCount) { // https://cp-algorithms.com/algebra/fft.html
    assert((sampleCount & (sampleCount - 1)) == 0); // power of 2

    if (sampleCount  == 1) {
        return { samples[0] };
    }

    auto even = std::vector<std::complex<float>>(sampleCount / 2);
    auto odd = std::vector<std::complex<float>>(sampleCount / 2);
    for (int i = 0; i < sampleCount / 2; i ++) {
        even[i] = samples[i * 2];
        odd[i] = samples[i * 2 + 1];
    }

    auto fftEven = recursive_fft(even, sampleCount / 2);
    auto fftOdd = recursive_fft(odd, sampleCount / 2);
    auto values = std::vector<std::complex<float>>(sampleCount);
    for (int k = 0; k < sampleCount / 2; k++) {
        values[k] = fftEven[k] + W_map[k] * fftOdd[k];
        values[k + sampleCount / 2] = fftEven[k] - W_map[k] * fftOdd[k];
    }

    return values;
}

int reverse(int x, int bits = 32) {
    int result = 0;
    for (int i = 0; i < bits; i++) {
        if ((x & (1 << i)) != 0) {
            result |= 1 << (bits - 1 - i);
        }
    }
    return result;
}

// https://edp.org/work/Construction.pdf
std::vector<std::complex<float>> fft_radix4(std::vector<std::complex<float>> samples, int sampleCount) { // https://cp-algorithms.com/algebra/fft.html
    assert(( (sampleCount & (-sampleCount)) & 0x55555554 ) == sampleCount);
    using namespace std::complex_literals;

    //bit_reverse_radix4(samples, sampleCount);

    for (int i = 0; i <= sampleCount / 4; i += 1) {
        //const std::complex<float> W = exp(-2.0if * (float) M_PI  / (float) i);
        const auto angle =  -2.0f * (float) M_PI / (float) i;
        for (int j = 0; j < sampleCount; j+=i) {
            auto len4 = i/4;
            for (int k = 0; k < len4; ++k) {
                //auto a = samples[j + k];
                /*auto b = samples[j + k + i/4] * pow(W, (float) k);
                auto c = samples[j + k + 2 * i/4] * pow(W, 2 * (float) k);
                auto d = samples[j + k + 3 * i/4] * pow(W, 3* (float) k);*/

                // Radix-4 butterfly

                auto angle1 = angle * (j);
                auto angle2 = angle * 2 * j;
                auto angle3 = angle * 3 * j;

                auto w1_re = cos(angle1);
                auto w1_im = sin(angle1);
                auto w2_re = cos(angle2);
                auto w2_im = sin(angle2);
                auto w3_re = cos(angle3);
                auto w3_im = sin(angle3);

                auto idx0 = i + j;
                auto idx1 = i + j + len4;
                auto idx2 = i + j + 2 * len4;
                auto idx3 = i + j + 3 * len4;

                auto a = samples[idx0];
                auto b = samples[idx1] * (w1_re + 1if * w1_im);
                auto c = samples[idx2] * (w2_re + 1if * w2_im);
                auto d = samples[idx3]* (w3_re + 1if * w3_im);


                auto sum = a + c;
                auto diff = a - c;
                auto sum_bd = b + d;
                auto diff_bd = (b-d) * -1.if;

                samples[idx0] = sum + sum_bd;
                samples[idx1] = diff + (diff_bd);
                samples[idx2] = sum - (sum_bd);
                samples[idx3] = diff - (diff_bd);
            }
        }
    }

    return samples;
}



void bit_reverse(std::vector<std::complex<float>>& samples, int sampleCount) {
    for (int i = 0; i < sampleCount; ++i) {
        if (i < reverse(i, log2(sampleCount))) {
            std::complex<float> tmp = samples[i];
            samples[i] = samples[reverse(i, log2(sampleCount))];
            samples[reverse(i, log2(sampleCount))] = tmp;
        }
    }
}

// radix 2
std::vector<std::complex<float>> fft(std::vector<std::complex<float>> samples, int sampleCount) { // https://cp-algorithms.com/algebra/fft.html
    assert((sampleCount & (sampleCount - 1)) == 0);

    bit_reverse(samples, sampleCount);

    for (int i = 2; i <= sampleCount; i <<= 1) {
        for (int j = 0; j < sampleCount; j+=i) {

            for (int k = 0; k < i/2; ++k) {
                auto t = samples[j + k];
                auto u =  W_map[k * (sampleCount / i)] * samples[j + k + i / 2];
                samples[j + k] = t + u;
                samples[j + k + i / 2] = u -t;
            }
        }
    }

    return samples;
}

// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
int main() {
    wav_header wavheader{};
    FILE* file = fopen("/Users/ilcheese2/Downloads/If\ I\ could\ be\ a\ constellation\ \[TubeRipper.cc\].wav","rb");
    fread(&wavheader,sizeof(wav_header),1,file);

    int bytesPerSample = wavheader.bit_depth / 8;

    char chunkName[4] {};
    fread(&chunkName, 4, 1, file);

    while (strncmp(chunkName, "data", 4) != 0) {
        int chunkSize = 0;
        fread(&chunkSize, 4, 1, file);
        fseek(file, chunkSize, SEEK_CUR);
        fread(&chunkName, 4, 1, file);
    }

    int dataSize;
    fread(&dataSize, 4, 1, file);

    int sampleCount = dataSize / (wavheader.num_channels * bytesPerSample);

    assert(wavheader.audio_format == 1);
    assert(bytesPerSample == 2); // signed > 16 bit

    std::vector<float> samples = std::vector<float>(sampleCount);
    auto* complexSamples = new fftwf_complex[sampleCount];
    for (int i = 0; i < sampleCount; i++) {
        short sample;
        fread(&sample, bytesPerSample, 1, file);

        if (wavheader.num_channels == 2) fread(&sample, bytesPerSample, 1, file); // to mono

        samples[i] = (float) sample / 32768.f;
        complexSamples[i][0] = samples[i];
        complexSamples[i][1] = 0.f;
    }


    fclose(file);
    int a = 0;
    while (a < sampleCount) {
        if (samples[a] != 0.f) {
            break;
        }
        a++;
    }

    int N = 15;
    sampleCount = 1 << N;

    samples = std::vector(samples.begin()+a,samples.begin()+a+sampleCount);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::vector<float> imag = std::vector(sampleCount, 0.f);

    int iterations = 40;
    for (int i = 0; i < iterations; i++) {
        do_fft(&samples[0], &imag[0], sampleCount);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "avg fft time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/iterations << "[µs]" << std::endl;

    complexSamples += a;

    fftwf_complex *in, *out;
    fftwf_plan p;

    in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * sampleCount);
    out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * sampleCount);
    memcpy(in, complexSamples, sizeof(fftwf_complex) * sampleCount);
    begin = std::chrono::steady_clock::now();
    p = fftwf_plan_dft_1d(sampleCount, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftwf_execute(p); /* repeat as needed */
    end = std::chrono::steady_clock::now();
    std::cout << "fftw time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    for (int i = 0; i < sampleCount; ++i) {
        //std::cout << "ours " <<  samples2[i] << ", " << imag[i] << " fftw " << out[i][0] << ", " << out[i][1] << "\n";
        if (fabs(out[i][0] - samples[i]) > 0.01f || fabs(out[i][1] - imag[i]) > 0.01f) {
            std::cout << "mismatch at " << i << " ours " <<  samples[i] << ", " << imag[i] << " fftw " << out[i][0] << ", " << out[i][1] << "\n";
        }
    }

    fftwf_destroy_plan(p);

    fftwf_free(in); fftwf_free(out);
}