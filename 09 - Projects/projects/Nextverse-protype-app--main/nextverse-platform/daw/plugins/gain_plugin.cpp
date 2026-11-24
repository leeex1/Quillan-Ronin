#include "plugin_api.h"
#include <cmath>
#include <cstring>
#include <string>

static double g_gain = 0.5;

extern "C" {

const char* nv_plugin_name(void) {
    return "NextVerse Native Gain";
}

int nv_process(const float* input, float* output, int frames,
               int channels, double sample_rate) {
    (void)sample_rate;
    if (input == nullptr || output == nullptr || frames <= 0 || channels <= 0) {
        return 0;
    }
    const int total = frames * channels;
    for (int i = 0; i < total; ++i) {
        output[i] = input[i] * static_cast<float>(g_gain);
    }
    return 1;
}

void nv_set_parameter(const char* name, double value) {
    if (name != nullptr && std::string(name) == "gain" && value >= 0.0 && value <= 2.0) {
        g_gain = value;
    }
}

}
