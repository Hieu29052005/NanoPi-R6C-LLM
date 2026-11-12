from __future__ import annotations

import ctypes
import logging
import os
import threading
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

RKLLM_Handle_t = ctypes.c_void_p

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL = 0
LLMCallState.RKLLM_RUN_WAITING = 1
LLMCallState.RKLLM_RUN_FINISH = 2
LLMCallState.RKLLM_RUN_ERROR = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT = 0
RKLLMInputType.RKLLM_INPUT_TOKEN = 1
RKLLMInputType.RKLLM_INPUT_EMBED = 2
RKLLMInputType.RKLLM_INPUT_MULTIMODAL = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("ignore_eos_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMImageInput(ctypes.Structure):
    _fields_ = [
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_start", ctypes.c_char_p),
        ("image_end", ctypes.c_char_p),
        ("image_content", ctypes.c_char_p),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMVideoInput(ctypes.Structure):
    _fields_ = [
        ("video_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_frame_tokens", ctypes.c_size_t),
        ("n_frame_per_video", ctypes.c_size_t),
        ("n_video", ctypes.c_size_t),
        ("video_start", ctypes.c_char_p),
        ("video_end", ctypes.c_char_p),
        ("video_content", ctypes.c_char_p),
        ("frame_width", ctypes.c_size_t),
        ("frame_height", ctypes.c_size_t),
    ]


class RKLLMAudioInput(ctypes.Structure):
    _fields_ = [
        ("audio_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_audio_tokens", ctypes.c_size_t),
        ("n_audio", ctypes.c_size_t),
        ("audio_start", ctypes.c_char_p),
        ("audio_end", ctypes.c_char_p),
        ("audio_content", ctypes.c_char_p),
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image", RKLLMImageInput),
        ("video", RKLLMVideoInput),
        ("audio", RKLLMAudioInput),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(ctypes.Structure):
    _anonymous_ = ("input_data",)
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMSamplingParam(ctypes.Structure):
    _fields_ = [
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("sampling_params", ctypes.POINTER(RKLLMSamplingParam)),
        ("keep_history", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int32),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


LLMResultCallback_type = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)
LLMTokenizerCallback_type = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32
)
LLMGetEmbedCallback_type = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64
)


class RKLLMCallback(ctypes.Structure):
    _fields_ = [
        ("result_callback", LLMResultCallback_type),
        ("result_userdata", ctypes.c_void_p),
        ("tokenizer_callback", LLMTokenizerCallback_type),
        ("tokenizer_userdata", ctypes.c_void_p),
        ("embed_callback", LLMGetEmbedCallback_type),
        ("embed_userdata", ctypes.c_void_p),
    ]


class RKLLMRuntime:
    def __init__(
        self,
        library_path: str,
        model_path: str,
        target_platform: str = "rk3588",
        max_context: int = 2048,
        max_new_tokens: int = 192,
        temperature: float = 0.3,
        top_k: int = 1,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        enable_thinking: bool = False,
        num_npu_core: int = 3,
        system_prompt: str = "",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"rkllm model not found: {model_path}")
        if not os.path.exists(library_path):
            raise FileNotFoundError(f"librkllmrt.so not found: {library_path}")

        self.library_path = library_path
        self.model_path = model_path
        self.enable_thinking = bool(enable_thinking)
        self.system_prompt = system_prompt or ""
        self._lock = threading.Lock()
        self._chunks: List[bytes] = []
        self._state = -1
        self._perf: dict = {}

        self.lib = ctypes.CDLL(library_path)

        self._callback = LLMResultCallback_type(self._on_result)

        param = RKLLMParam()
        ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(RKLLMParam))
        param.model_path = model_path.encode("utf-8")
        param.max_context_len = int(max_context)
        param.max_new_tokens = int(max_new_tokens)
        param.top_k = int(top_k)
        param.n_keep = -1
        param.top_p = float(top_p)
        param.temperature = float(temperature)
        param.repeat_penalty = float(repeat_penalty)
        param.frequency_penalty = 0.0
        param.presence_penalty = 0.0
        param.mirostat = 0
        param.mirostat_tau = 5.0
        param.mirostat_eta = 0.1
        param.skip_special_token = True
        param.ignore_eos_token = False
        param.is_async = False
        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1
        param.extend_param.n_batch = 1
        param.extend_param.use_cross_attn = 0
        param.extend_param.enabled_cpus_num = 4
        if target_platform.lower() in ("rk3576", "rk3588", "rk3588s"):
            param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        else:
            param.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

        self.handle = RKLLM_Handle_t()
        callback = RKLLMCallback()
        callback.result_callback = self._callback
        callback.result_userdata = None
        callback.tokenizer_callback = LLMTokenizerCallback_type()
        callback.tokenizer_userdata = None
        callback.embed_callback = LLMGetEmbedCallback_type()
        callback.embed_userdata = None

        init = self.lib.rkllm_init
        init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), ctypes.POINTER(RKLLMCallback)]
        init.restype = ctypes.c_int
        ret = init(ctypes.byref(self.handle), ctypes.byref(param), ctypes.byref(callback))
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed with code {ret}")

        self._run = self.lib.rkllm_run
        self._run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self._run.restype = ctypes.c_int

        self._abort = self.lib.rkllm_abort
        self._abort.argtypes = [RKLLM_Handle_t]
        self._abort.restype = ctypes.c_int

        self._destroy = self.lib.rkllm_destroy
        self._destroy.argtypes = [RKLLM_Handle_t]
        self._destroy.restype = ctypes.c_int

        self._set_chat_template = self.lib.rkllm_set_chat_template
        self._set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self._set_chat_template.restype = ctypes.c_int

        self._infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self._infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self._infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self._infer_params.keep_history = 0

        self.model_name = os.path.basename(model_path).rsplit(".", 1)[0]
        self._closed = False
        log.info("RKLLM runtime ready: %s", self.model_name)

    def _on_result(self, result, userdata, state):
        try:
            if state == LLMCallState.RKLLM_RUN_NORMAL:
                if result and result.contents.text:
                    self._chunks.append(result.contents.text)
            elif state == LLMCallState.RKLLM_RUN_WAITING:
                pass
            elif state == LLMCallState.RKLLM_RUN_FINISH:
                self._state = state
                if result:
                    perf = result.contents.perf
                    self._perf = {
                        "prefill_time_ms": float(perf.prefill_time_ms),
                        "prefill_tokens": int(perf.prefill_tokens),
                        "generate_time_ms": float(perf.generate_time_ms),
                        "generate_tokens": int(perf.generate_tokens),
                        "memory_usage_mb": float(perf.memory_usage_mb),
                    }
            elif state == LLMCallState.RKLLM_RUN_ERROR:
                self._state = state
            else:
                self._state = state
        except Exception as exc:
            log.error("rkllm callback error: %s", exc)
        return 0

    def set_chat_template(self, system_prompt: str, prefix: str = "", postfix: str = "") -> None:
        self._set_chat_template(
            self.handle,
            system_prompt.encode("utf-8"),
            prefix.encode("utf-8"),
            postfix.encode("utf-8"),
        )

    def _run_once(self, prompt: str, max_new_tokens: Optional[int]) -> None:
        rkllm_input = RKLLMInput()
        rkllm_input.role = b"user"
        rkllm_input.enable_thinking = self.enable_thinking
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.prompt_input = prompt.encode("utf-8")

        if max_new_tokens is not None:
            self._infer_params.max_new_tokens = int(max_new_tokens)
        self._run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self._infer_params), None)

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, timeout: Optional[float] = None) -> Tuple[str, dict]:
        if self._closed:
            raise RuntimeError("RKLLM runtime is closed")
        with self._lock:
            self._chunks = []
            self._state = -1
            self._perf = {}
            worker = threading.Thread(target=self._run_once, args=(prompt, max_new_tokens), daemon=True)
            worker.start()
            worker.join(timeout=timeout)
            if worker.is_alive():
                log.warning("rkllm generation timed out after %s s, aborting", timeout)
                self._abort(self.handle)
                worker.join(timeout=5.0)
            if self._state == LLMCallState.RKLLM_RUN_ERROR:
                raise RuntimeError("rkllm generation returned an error state")
            raw = b"".join(self._chunks)
            text = raw.decode("utf-8", errors="replace").strip()
            if max_new_tokens is not None:
                self._infer_params.max_new_tokens = 0
            return text, dict(self._perf)

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._destroy(self.handle)
        except Exception as exc:
            log.warning("rkllm_destroy failed: %s", exc)
