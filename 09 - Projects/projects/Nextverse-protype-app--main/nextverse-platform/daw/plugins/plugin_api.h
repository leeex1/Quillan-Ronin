#ifndef NEXVERSE_PLUGIN_API_H
#define NEXVERSE_PLUGIN_API_H

/* Minimal native plugin ABI for the NextVerse DAW host.
   Not VST3/CLAP: a locally defined, testable C ABI built with the host's
   existing compiler. All functions use C linkage and plain C types. */

#ifdef _WIN32
#define NV_EXPORT __declspec(dllexport)
#else
#define NV_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

NV_EXPORT const char* nv_plugin_name(void);
NV_EXPORT int nv_process(const float* input, float* output, int frames,
                         int channels, double sample_rate);
NV_EXPORT void nv_set_parameter(const char* name, double value);

#ifdef __cplusplus
}
#endif
#endif
