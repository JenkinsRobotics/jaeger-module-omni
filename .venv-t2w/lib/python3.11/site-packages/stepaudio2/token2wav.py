import io
import sys
import types

import torch
import torchaudio
import s3tokenizer
import onnxruntime
import numpy as np
from copy import deepcopy

import torchaudio.compliance.kaldi as kaldi
from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator
from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram
from hyperpyyaml import load_hyperpyyaml


def _setup_cosyvoice2_alias():
    """给 hyperpyyaml 提供 cosyvoice2.* 的兼容别名，不改 flow.yaml。"""
    if 'cosyvoice2.flow.flow' in sys.modules:
        # 已经设置过，直接复用
        return

    # 导入 stepaudio2 里真实的实现
    import stepaudio2.cosyvoice2.flow.flow as _step_flow
    import stepaudio2.cosyvoice2.flow.flow_matching as _step_flow_matching
    import stepaudio2.cosyvoice2.flow.decoder_dit as _step_decoder_dit
    import stepaudio2.cosyvoice2.transformer.upsample_encoder_v2 as _step_upsample

    # 创建顶层 cosyvoice2 包和子包
    cosyvoice2_pkg = types.ModuleType('cosyvoice2')
    cosyvoice2_flow_pkg = types.ModuleType('cosyvoice2.flow')
    cosyvoice2_transformer_pkg = types.ModuleType('cosyvoice2.transformer')

    # 挂载子模块
    cosyvoice2_flow_pkg.flow = _step_flow
    cosyvoice2_flow_pkg.flow_matching = _step_flow_matching
    cosyvoice2_flow_pkg.decoder_dit = _step_decoder_dit

    cosyvoice2_transformer_pkg.upsample_encoder_v2 = _step_upsample

    cosyvoice2_pkg.flow = cosyvoice2_flow_pkg
    cosyvoice2_pkg.transformer = cosyvoice2_transformer_pkg

    # 注册到 sys.modules，让 `cosyvoice2.flow.flow.*` 这类路径可被 import
    sys.modules['cosyvoice2'] = cosyvoice2_pkg
    sys.modules['cosyvoice2.flow'] = cosyvoice2_flow_pkg
    sys.modules['cosyvoice2.flow.flow'] = _step_flow
    sys.modules['cosyvoice2.flow.flow_matching'] = _step_flow_matching
    sys.modules['cosyvoice2.flow.decoder_dit'] = _step_decoder_dit
    sys.modules['cosyvoice2.transformer'] = cosyvoice2_transformer_pkg
    sys.modules['cosyvoice2.transformer.upsample_encoder_v2'] = _step_upsample

def fade_in_out(fade_in_mel:torch.Tensor, fade_out_mel:torch.Tensor, window:torch.Tensor):
    """perform fade_in_out in tensor style
    """
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = \
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel


class Token2wav():

    def __init__(self, model_path, float16=False, n_timesteps=10):
        self.float16 = float16
        self.n_timesteps = n_timesteps
        # 在加载 flow.yaml 之前，先把 cosyvoice2 的别名注册好
        _setup_cosyvoice2_alias()

        self.audio_tokenizer = s3tokenizer.load_model(f"{model_path}/speech_tokenizer_v2_25hz.onnx").cuda().eval()

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(f"{model_path}/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"])

        with open(f"{model_path}/flow.yaml", "r") as f:
            configs = load_hyperpyyaml(f)
            self.flow = configs['flow']
        if float16:
            self.flow.half()
        self.flow.load_state_dict(torch.load(f"{model_path}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        self.flow.cuda().eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{model_path}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.cuda().eval()

        self.cache = None

        # stream conf
        self.mel_cache_len = 8  # hard-coded, 160ms
        self.source_cache_len = int(self.mel_cache_len * 480)   # 50hz mel -> 24kHz wave
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).cuda()

        # hifigan cache
        self.hift_cache_dict = {}


    def _prepare_prompt(self, prompt_wav):
        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)  # [T]
        # TODO 在audio 后面 pad 3个token长度 = 0.12s的音频
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(mels.cuda(), mels_lens.cuda())
        # TODO 手动赋值最后3个token为 4218 静音token

        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(self.spk_model.run(
            None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
        )[0], device='cuda')

        audio, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
        audio = audio.mean(dim=0, keepdim=True)  # [1, T]
        if sample_rate != 24000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
        prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
        prompt_mels = prompt_mel.unsqueeze(0).cuda()
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device='cuda')
        prompt_mels = torch.nn.functional.pad(prompt_mels, (0, 0, 0, prompt_speech_tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]), mode='replicate')
        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    def __call__(self, generated_speech_tokens, prompt_wav):
        if self.cache is None:
            self.cache = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device='cuda')
        generated_speech_tokens_lens = torch.tensor([generated_speech_tokens.shape[1]], dtype=torch.int32, device='cuda')

        with torch.amp.autocast("cuda", dtype=torch.float16 if self.float16 else torch.float32):
            mel = self.flow.inference(generated_speech_tokens, generated_speech_tokens_lens,
                prompt_speech_tokens, prompt_speech_tokens_lens,
                prompt_mels, prompt_mels_lens, spk_emb, 10)

        wav, _ = self.hift(speech_feat=mel)
        output = io.BytesIO()
        torchaudio.save(output, wav.cpu(), sample_rate=24000, format='wav')

        return output.getvalue()

    def set_stream_cache(self, prompt_wav):
        self.cache = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache

        print("prompt_speech_tokens_lens", prompt_speech_tokens_lens)
        print("prompt_mels.shape", prompt_mels.shape)
        print("prompt_mels_lens", prompt_mels_lens)
        
        right_pad_speech_tokens = torch.ones(1, 3, device=prompt_speech_tokens.device, dtype=prompt_speech_tokens.dtype) * 4218
        
        # self.stream_cache = self.flow.setup_cache(
        #     torch.cat([prompt_speech_tokens, prompt_speech_tokens[:, :3]], dim=1),
        #     prompt_mels, spk_emb, n_timesteps=10)
        
        stream_cache = self.flow.setup_cache(
            torch.cat([prompt_speech_tokens, right_pad_speech_tokens], dim=1),
            prompt_mels, spk_emb, n_timesteps=self.n_timesteps)

        # hift cache
        hift_cache_dict = dict(
            mel = torch.zeros(1, prompt_mels.shape[2], 0, device='cuda'), 
            source = torch.zeros(1, 1, 0, device='cuda'),
            speech = torch.zeros(1, 0, device='cuda'),
        )

        return stream_cache, hift_cache_dict


    def stream(self, generated_speech_tokens, prompt_wav, last_chunk=False, return_waveform=False):
        if self.cache is None:
            self.cache = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device='cuda')
        generated_speech_tokens_lens = torch.tensor([generated_speech_tokens.shape[1]], dtype=torch.int32, device='cuda')

        if self.stream_cache is None:
            raise ValueError("stream_cache is not set")

        with torch.amp.autocast("cuda", dtype=torch.float16 if self.float16 else torch.float32):
            chunk_mel, self.stream_cache = self.flow.inference_chunk(
                token=generated_speech_tokens,
                spk=spk_emb,
                cache=self.stream_cache,
                last_chunk=last_chunk,
                n_timesteps=self.n_timesteps,
            )
        if self.stream_cache['estimator_att_cache'].shape[4] > (prompt_mels.shape[1] + 100):
            self.stream_cache['estimator_att_cache'] = torch.cat([
                self.stream_cache['estimator_att_cache'][:, :, :, :, :prompt_mels.shape[1]],
                self.stream_cache['estimator_att_cache'][:, :, :, :, -100:],
            ], dim=4)

        # bug fix - 20260107
        # 同样截断 conformer_att_cache，防止无限增长导致 position embedding 不匹配
        if self.stream_cache['conformer_att_cache'].shape[3] > (prompt_mels.shape[1] + 100):
            self.stream_cache['conformer_att_cache'] = torch.cat([
                self.stream_cache['conformer_att_cache'][:, :, :, :prompt_mels.shape[1], :],
                self.stream_cache['conformer_att_cache'][:, :, :, -100:, :],
            ], dim=3)
        
        # vocoder cache
        hift_cache_mel = self.hift_cache_dict['mel']
        hift_cache_source = self.hift_cache_dict['source']
        hift_cache_speech = self.hift_cache_dict['speech']
        mel = torch.concat([hift_cache_mel, chunk_mel], dim=2)

        speech, source = self.hift(mel, hift_cache_source)

        # overlap speech smooth
        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)

        # update vocoder cache
        self.hift_cache_dict = dict(
            mel = mel[..., -self.mel_cache_len:].clone().detach(),
            source = source[:, :, -self.source_cache_len:].clone().detach(),
            speech = speech[:, -self.source_cache_len:].clone().detach(),
        )
        if not last_chunk:
            speech = speech[:, :-self.source_cache_len]

        wav_np = speech.cpu().numpy()
        if return_waveform:
            return wav_np
        
        # Clip to [-1, 1] to avoid overflow, then scale to int16
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767.0).astype('<i2')  # 16-bit little-endian PCM
        pcm_bytes = wav_int16.tobytes()
        return pcm_bytes

