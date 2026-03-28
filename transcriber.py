import os
import traceback
import whisper
import torch

# ---------------------------------------------------------------------------
# Module-level model caches.
# ---------------------------------------------------------------------------
_whisper_models = {}
_hinglish_pipeline = None


def _get_whisper_model(model_size: str):
    if model_size not in _whisper_models:
        print(f"[Transcriber] Loading Whisper model: {model_size} ...")
        _whisper_models[model_size] = whisper.load_model(model_size)
        print(f"[Transcriber] Whisper '{model_size}' ready.")
    return _whisper_models[model_size]


def _get_hinglish_pipeline():
    global _hinglish_pipeline
    if _hinglish_pipeline is None:
        from transformers import (
            WhisperForConditionalGeneration,
            AutoProcessor,
            pipeline as hf_pipeline,
        )

        model_id = "Oriserve/Whisper-Hindi2Hinglish-Apex"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[Transcriber] Loading Hindi->Hinglish model on {device} ...")

        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map=device
        )

        # ---------------------------------------------------------------
        # FIX 1: Clear suppress_tokens duplication.
        # This model stores them in BOTH model.config and generation_config.
        # Whisper's .generate() auto-creates SuppressTokensLogitsProcessor
        # from generation_config — having them in model.config too causes a
        # duplicate conflict crash. Clear from both; the pipeline re-derives
        # them from task/language at call time.
        # ---------------------------------------------------------------
        model.config.suppress_tokens = []
        model.config.begin_suppress_tokens = []
        if hasattr(model.generation_config, "suppress_tokens"):
            model.generation_config.suppress_tokens = None
        if hasattr(model.generation_config, "begin_suppress_tokens"):
            model.generation_config.begin_suppress_tokens = None

        # ---------------------------------------------------------------
        # FIX 2: Set alignment_heads manually.
        # return_timestamps="word" needs alignment_heads to locate which
        # cross-attention heads map audio frames to tokens. This fine-tuned
        # model doesn't save them in its config (they come back as None),
        # causing: TypeError: 'NoneType' object is not iterable
        # Solution: inject the standard Whisper-medium alignment heads,
        # which this model is fine-tuned from.
        # Source: https://github.com/openai/whisper/blob/main/whisper/timing.py
        # ---------------------------------------------------------------
        # Inspect the actual model dimensions so we always inject valid heads
        # regardless of which Whisper variant this fine-tune is based on.
        num_decoder_layers = model.config.decoder_layers
        num_heads          = model.config.decoder_attention_heads
        print(f"[Transcriber] Model dims: decoder_layers={num_decoder_layers}, attention_heads={num_heads}")

        # All known Whisper alignment_heads keyed by (decoder_layers, heads).
        # Source: openai/whisper timing.py + HuggingFace model cards.
        WHISPER_ALIGNMENT_HEADS = {
            (4,  6):  [[1, 1], [2, 0], [2, 5], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]],                         # tiny
            (6,  8):  [[3, 0], [4, 7], [5, 1], [5, 5]],                                                           # base
            (12, 12): [[0, 0], [3, 0], [6, 0], [8, 0], [9, 8], [9, 10], [10, 0], [11, 3]],                       # small
            (24, 16): [[5, 3], [5, 9], [8, 0], [8, 4], [8, 7], [8, 8], [9, 0], [9, 7], [9, 9], [10, 5]],        # medium
            (32, 20): [[10,12],[13,17],[16,11],[16,12],[16,13],[17,15],[17,16],[18, 4],  # large / large-v1
                       [18,11],[18,19],[19,11],[21, 2],[21, 3],[22, 3],[22, 9],
                       [22,12],[23, 5],[23, 7],[23,13],[25, 5],[26, 1],[26,12],[27,15]],
        }

        key = (num_decoder_layers, num_heads)
        if key in WHISPER_ALIGNMENT_HEADS:
            heads = WHISPER_ALIGNMENT_HEADS[key]
            print(f"[Transcriber] Found known alignment_heads for {key}: {len(heads)} heads.")
            model.generation_config.alignment_heads = heads
        else:
            # Unknown architecture — use ALL heads across ALL layers.
            # This is the most accurate option for an unrecognised fine-tune:
            # rather than guessing which heads are "best", we let every head
            # contribute to the cross-attention average. The timestamp quality
            # is slightly lower than hand-picked heads but far better than
            # using only a single layer (our old fallback), and it is
            # guaranteed to be in-bounds for any architecture.
            heads = [[l, h] for l in range(num_decoder_layers) for h in range(num_heads)]
            print(f"[Transcriber] Unknown architecture {key} — using ALL {len(heads)} heads across all layers.")
            model.generation_config.alignment_heads = heads

        print(f"[Transcriber] alignment_heads set: {len(model.generation_config.alignment_heads)} heads.")

        processor = AutoProcessor.from_pretrained(model_id)

        # return_timestamps intentionally omitted from constructor —
        # passed at call time to avoid generation_config conflict.
        _hinglish_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=device,
            generate_kwargs={"task": "transcribe", "language": "en"},
            chunk_length_s=30,
        )
        print("[Transcriber] Hindi->Hinglish model ready.")

    return _hinglish_pipeline


def _detect_language(video_path: str) -> str:
    model = _get_whisper_model("base")
    audio = whisper.load_audio(video_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected = max(probs, key=probs.get)
    confidence = probs[detected]
    print(f"[Transcriber] Detected language: '{detected}' (confidence: {confidence:.1%})")
    return detected


def _normalise_hinglish_segments(result: dict, max_words: int = 6, max_duration: float = 3.5, silence_gap: float = 0.4) -> list:
    """
    Groups word-level chunks into subtitle segments with three flush triggers:
      1. max_words reached
      2. max_duration reached
      3. silence gap detected (next word starts more than silence_gap seconds
         after the previous word ended) — this is the key fix for the
         "first word of next sentence shown during silence" bug.

    After grouping, each segment's end time is clamped to just before the
    next segment's start, so no subtitle bleeds into a silence gap.
    """
    chunks = result.get("chunks", [])
    print(f"[Transcriber] _normalise: received {len(chunks)} raw chunks.")

    segments = []
    seg_id = 0
    current_words = []
    current_word_data = []
    seg_start = None
    seg_end = None

    def flush():
        nonlocal seg_id, seg_start, seg_end, current_words, current_word_data
        if not current_words:
            return
        segments.append({
            "id":    seg_id,
            "text":  " ".join(current_words),
            "start": seg_start,
            "end":   seg_end,
            "words": current_word_data,
        })
        seg_id += 1
        current_words = []
        current_word_data = []
        seg_start = None
        seg_end = None

    for chunk in chunks:
        word_text = chunk.get("text", "").strip()
        if not word_text:
            continue

        timestamp = chunk.get("timestamp") or (None, None)
        w_start = timestamp[0] if timestamp[0] is not None else (seg_end or 0.0)
        w_end   = timestamp[1] if timestamp[1] is not None else w_start + 0.3

        if seg_start is None:
            seg_start = w_start

        duration_so_far = w_end - seg_start

        # --- Three flush triggers ---
        # 1. Silence gap: there is a pause before this word — flush the
        #    current segment so nothing shows during the silence.
        gap = w_start - seg_end if seg_end is not None else 0.0
        silence_detected = current_words and gap >= silence_gap

        # 2 & 3. Too many words or segment too long
        limits_reached = current_words and (
            len(current_words) >= max_words or duration_so_far > max_duration
        )

        if silence_detected or limits_reached:
            flush()
            seg_start = w_start

        current_words.append(word_text)
        current_word_data.append({"word": word_text, "start": w_start, "end": w_end})
        seg_end = w_end

    flush()

    # --- Pin every segment's end to its last word's actual end time ---
    # This is the core fix for subtitles showing during silence.
    # The pipeline's chunk timestamps only tell us when each word starts/ends.
    # A segment's end must ALWAYS equal its last word's end — never bleed
    # beyond it into silence, regardless of when the next segment starts.
    for seg in segments:
        if seg["words"]:
            seg["end"] = seg["words"][-1]["end"]

    # --- Additionally clamp to avoid overlap with next segment ---
    # If two segments are back-to-back with no gap, trim the earlier one
    # to 50ms before the next starts so they never overlap on screen.
    CLAMP_MARGIN = 0.05  # 50 ms breathing room
    for i in range(len(segments) - 1):
        next_start = segments[i + 1]["start"]
        if segments[i]["end"] > next_start - CLAMP_MARGIN:
            segments[i]["end"] = max(segments[i]["start"] + 0.1, next_start - CLAMP_MARGIN)
            if segments[i]["words"]:
                segments[i]["words"][-1]["end"] = segments[i]["end"]

    print(f"[Transcriber] _normalise: produced {len(segments)} segments.")
    return segments


def transcribe_video(video_path: str, model_size: str = "medium") -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    lang = _detect_language(video_path)

    if lang == "hi":
        print("[Transcriber] Hindi detected — routing to Hinglish model.")
        try:
            pipe = _get_hinglish_pipeline()
            print("[Transcriber] Running inference...")
            raw_result = pipe(video_path, return_timestamps="word")
            print(f"[Transcriber] Inference done. Text preview: {str(raw_result.get('text',''))[:200]}")
            segments = _normalise_hinglish_segments(raw_result)
            return {"segments": segments, "language": "hi"}
        except Exception:
            print("[Transcriber] FAILED — full traceback:")
            traceback.print_exc()
            raise

    else:
        print(f"[Transcriber] Language '{lang}' — using standard Whisper '{model_size}'.")
        model = _get_whisper_model(model_size)
        result = model.transcribe(video_path, word_timestamps=True)
        return {"segments": result["segments"], "language": result["language"]}
