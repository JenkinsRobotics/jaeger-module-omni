// Research-only bridge: configure the actual audio prompt, not the separate
// omni/video prompt exposed by the older local shim. Never changes model weights.
#include "omni.h"
#include <string>

extern "C" void paired_c_audio_instruction(void * context, const char * text, int user_task) {
    auto * o = static_cast<omni_context *>(context);
    // Match the engine's documented no-TTS chat path: a plain system message,
    // no voice-clone reference slots, and native audio markers around the input.
    o->audio_voice_clone_prompt = std::string("<|im_start|>system\n") + text;
    o->audio_assistant_prompt = "<|im_end|>\n<|im_start|>user\n";
    if (user_task) {
        o->audio_voice_clone_prompt = "<|im_start|>system\nYou are a helpful assistant.";
        o->audio_assistant_prompt += std::string(text) + "\n";
    }
    o->system_prompt_initialized = false;
}

extern "C" const char * paired_c_audio_prompt(void * context) {
    static thread_local std::string prompt;
    auto * o = static_cast<omni_context *>(context);
    prompt = o->audio_voice_clone_prompt + o->audio_assistant_prompt;
    return prompt.c_str();
}
