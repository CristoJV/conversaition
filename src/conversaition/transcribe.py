#!/usr/bin/env python3

"""
Local speaker-attributed transcription using:

- nvidia/Nemotron-3-Diarization
- nvidia/nemotron-3.5-asr-streaming-0.6b

Everything runs locally. Models must already exist as .nemo files.

Input requirements:
    - WAV
    - mono
    - 16 kHz

Example:

    uv run python transcribe.py \
        -i conversation.wav \
        --diar-model models/Nemotron-3-Diarization.nemo \
        --asr-model models/nemotron-3.5-asr-streaming-0.6b.nemo \
        -o transcription.json
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Disable Hugging Face network access BEFORE importing NeMo.
# ----------------------------------------------------------------------
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ----------------------------------------------------------------------

import argparse
import json
from pathlib import Path

import nemo.collections.asr as nemo_asr
import soundfile as sf
import torch
from nemo.collections.asr.models.sortformer_diar_models import (
    SortformerEncLabelModel,
)
from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
    SpeakerTaggedASR,
    configure_diar_streaming,
    validate_feature_frame_strides,
)
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
)
from omegaconf import OmegaConf

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def build_config(
    audio_file: Path,
    *,
    language: str,
    max_speakers: int,
    device: str,
):
    """
    Minimal configuration expected by SpeakerTaggedASR.

    Values are based on NVIDIA's recommended configuration for
    Nemotron-3-Diarization + Nemotron-3.5-ASR-Streaming-0.6B.
    """

    return OmegaConf.create(
        {
            # Input
            "audio_file": str(audio_file),
            "manifest_file": None,
            # Device
            "device": device,
            # We are processing a normal audio file, not a live deployment.
            "deploy_mode": False,
            # Diarization
            "streaming_mode": True,
            "max_num_of_spks": max_speakers,
            "spkcache_len": None,
            "spkcache_update_period": 222,
            "fifo_len": 264,
            "diar_right_context": 0,
            "spk_supervision": "diar",
            # ASR
            "batch_size": 1,
            "parallel_speaker_strategy": True,
            # Nemotron 3.5 is a conventional ASR.
            # Speaker activity is used to mask each ASR stream.
            "masked_asr": True,
            "mask_preencode": False,
            # Only process currently detected speakers.
            "cache_gating": True,
            "cache_gating_buffer_size": 2,
            "single_speaker_mode": False,
            "binary_diar_preds": True,
            # Recommended Nemotron 3.5 streaming context.
            "att_context_size": [56, 13],
            # Language prompt
            "target_lang": language,
            "strip_lang_tags": True,
            "lang_tag_pattern": None,
            # Precision
            "use_amp": True,
            "precision": "bf16",
            # Streaming buffer
            "online_normalization": False,
            "pad_and_drop_preencoded": False,
            # Transcript generation
            "feat_len_sec": 0.01,
            "discarded_frames": 8,
            "word_window": 50,
            "sent_break_sec": 1.0,
            "fix_prev_words_count": 5,
            "update_prev_words_sentence": 5,
            "left_frame_shift": -1,
            "right_frame_shift": 0,
            "min_sigmoid_val": 1e-2,
            "ignored_initial_frame_steps": 5,
            # Disable NVIDIA example output helpers.
            "generate_realtime_scripts": False,
            "print_sample_indices": [0],
            "colored_text": False,
            "verbose": False,
            "print_time": False,
            "log": False,
        }
    )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def validate_audio(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {path}")

    info = sf.info(path)

    if info.samplerate != 16_000:
        raise ValueError(
            f"Expected 16 kHz audio, got {info.samplerate} Hz.\n"
            f"Convert with:\n"
            f"  ffmpeg -i {path} -ac 1 -ar 16000 output.wav"
        )

    if info.channels != 1:
        raise ValueError(
            f"Expected mono audio, got {info.channels} channels.\n"
            f"Convert with:\n"
            f"  ffmpeg -i {path} -ac 1 -ar 16000 output.wav"
        )


def validate_model(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} model not found: {path}")

    if path.suffix != ".nemo":
        raise ValueError(f"{name} model must be a local .nemo file: {path}")


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------


def load_models(
    asr_path: Path,
    diar_path: Path,
    cfg,
    device: torch.device,
):
    print(f"Loading diarization model: {diar_path}")

    diar_model = SortformerEncLabelModel.restore_from(
        restore_path=str(diar_path),
        map_location=device,
    )

    print(f"Loading ASR model: {asr_path}")

    asr_model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(asr_path),
        map_location=device,
    )

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    asr_model = asr_model.to(device)
    asr_model.eval()

    # Recommended context for Nemotron 3.5 ASR.
    if not hasattr(asr_model.encoder, "set_default_att_context_size"):
        raise RuntimeError("ASR model does not support streaming attention contexts")

    asr_model.encoder.set_default_att_context_size(
        att_context_size=cfg.att_context_size
    )

    # Language prompt supported by Nemotron 3.5.
    if hasattr(asr_model, "set_inference_prompt"):
        asr_model.set_inference_prompt(cfg.target_lang)

        if hasattr(asr_model, "decoding") and hasattr(
            asr_model.decoding,
            "set_strip_lang_tags",
        ):
            asr_model.decoding.set_strip_lang_tags(
                cfg.strip_lang_tags,
                lang_tag_pattern=cfg.lang_tag_pattern,
            )

    # ------------------------------------------------------------------
    # Diarization
    # ------------------------------------------------------------------

    use_bf16 = device.type == "cuda" and cfg.use_amp and torch.cuda.is_bf16_supported()

    diar_dtype = torch.bfloat16 if use_bf16 else torch.float32

    diar_model = diar_model.to(
        device=device,
        dtype=diar_dtype,
    )
    diar_model.eval()

    # ------------------------------------------------------------------
    # Verify that ASR and diarization frame geometries match.
    # ------------------------------------------------------------------

    validate_feature_frame_strides(
        asr_model=asr_model,
        diar_model=diar_model,
    )

    streaming_cfg = asr_model.encoder.streaming_cfg

    diar_chunk_len = streaming_cfg.valid_out_len + streaming_cfg.cache_drop_size

    configure_diar_streaming(
        diar_model=diar_model,
        cfg=cfg,
        output_subsampling_factor=(asr_model.encoder.subsampling_factor),
        diar_chunk_len=diar_chunk_len,
    )

    # Some diarization architectures use FeatureStacking.
    if isinstance(
        diar_model.encoder.pre_encode,
        FeatureStacking,
    ):
        cfg.pad_and_drop_preencoded = True

    return asr_model, diar_model, use_bf16


# ----------------------------------------------------------------------
# Transcription
# ----------------------------------------------------------------------


def transcribe(
    audio_file: Path,
    *,
    asr_model,
    diar_model,
    cfg,
    device: torch.device,
    use_bf16: bool,
) -> list[dict]:
    """
    Run the coupled streaming diarization + ASR pipeline.
    """

    samples = [
        {
            "audio_filepath": str(audio_file),
        }
    ]

    # --------------------------------------------------------------
    # NeMo's cache-aware audio buffer.
    # --------------------------------------------------------------

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=cfg.online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )

    streaming_buffer.append_audio_file(
        audio_filepath=str(audio_file),
        stream_id=-1,
    )

    # --------------------------------------------------------------
    # Speaker-aware ASR engine.
    # --------------------------------------------------------------

    streamer = SpeakerTaggedASR(
        cfg,
        asr_model,
        diar_model,
    )

    diar_right_offset = cfg.diar_right_context * diar_model.encoder.subsampling_factor

    chunks = streaming_buffer.iter_with_right_context(diar_right_offset)

    # --------------------------------------------------------------
    # Streaming inference
    # --------------------------------------------------------------

    for step_num, (
        chunk_audio,
        chunk_lengths,
        diar_chunk_audio,
        diar_chunk_lengths,
    ) in enumerate(chunks):
        if step_num == 0 and not cfg.pad_and_drop_preencoded:
            drop_extra_pre_encoded = 0
        else:
            drop_extra_pre_encoded = (
                asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
            )

        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ),
        ):
            streamer.perform_parallel_streaming_stt_spk(
                step_num=step_num,
                chunk_audio=chunk_audio,
                chunk_lengths=chunk_lengths,
                diar_chunk_audio=diar_chunk_audio,
                diar_chunk_lengths=diar_chunk_lengths,
                is_buffer_empty=(streaming_buffer.is_buffer_empty()),
                drop_extra_pre_encoded=(drop_extra_pre_encoded),
            )

    # --------------------------------------------------------------
    # Convert internal speaker streams into speaker turns.
    # --------------------------------------------------------------

    seglst = streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)

    result = []

    for segment in seglst:
        text = segment["words"].strip()

        if not text:
            continue

        result.append(
            {
                "speaker": segment["speaker"],
                "start": float(segment["start_time"]),
                "end": float(segment["end_time"]),
                "text": text,
            }
        )

    result.sort(key=lambda segment: segment["start"])

    return result


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Local Nemotron diarization + transcription")
    )

    parser.add_argument(
        "-i",
        "--input-file",
        type=Path,
        required=True,
        help="16 kHz mono WAV file",
    )

    parser.add_argument(
        "--diar-model",
        type=Path,
        default=Path("models/diarization/Nemotron-3-Diarization.nemo"),
    )

    parser.add_argument(
        "--asr-model",
        type=Path,
        default=Path("models/asr/nemotron-3.5-asr-streaming-0.6b.nemo"),
    )

    parser.add_argument(
        "--language",
        default="es-ES",
        help="Language prompt, e.g. es-ES or auto",
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=4,
        help="Maximum number of speakers (1-8)",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device, default: cuda:0",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    audio_file = args.input_file.resolve()
    asr_path = args.asr_model.resolve()
    diar_path = args.diar_model.resolve()

    validate_audio(audio_file)
    validate_model(asr_path, "ASR")
    validate_model(diar_path, "Diarization")

    if not 1 <= args.max_speakers <= 8:
        raise ValueError("--max-speakers must be between 1 and 8")

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    torch.set_float32_matmul_precision("highest")

    cfg = build_config(
        audio_file,
        language=args.language,
        max_speakers=args.max_speakers,
        device=str(device),
    )

    asr_model, diar_model, use_bf16 = load_models(
        asr_path,
        diar_path,
        cfg,
        device,
    )

    print()
    print(f"Audio:     {audio_file}")
    print(f"Device:    {device}")
    print(f"Language:  {args.language}")
    print(f"Speakers:  <= {args.max_speakers}")
    print(f"BF16:      {use_bf16}")
    print()
    print("Transcribing...")
    print()

    segments = transcribe(
        audio_file,
        asr_model=asr_model,
        diar_model=diar_model,
        cfg=cfg,
        device=device,
        use_bf16=use_bf16,
    )

    # JSON output
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                segments,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
