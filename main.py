"""
Irodori-TTS ベースモデルを使用した Gradio GUI
"""

import os
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download
import torch
import torchaudio

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    get_cached_runtime,
    default_runtime_device,
)
from voice_files import (
    NO_VOICE_OPTION,
    VOICE_EXTENSIONS,
    _get_output_subdir,
    get_voice_file_choices,
)

# ── 設定 ──────────────────────────────────────────────
HF_REPO_ID = "Aratako/Irodori-TTS-500M-v3"
HF_FILENAME = "model.safetensors"

MODEL_DEVICE = default_runtime_device()
CODEC_DEVICE = default_runtime_device()
MODEL_PRECISION = "bf16" if MODEL_DEVICE == "cuda" else "fp32"
CODEC_PRECISION = "bf16" if CODEC_DEVICE == "cuda" else "fp32"

OUT_DIR = Path(os.environ.get("FLEXITALK_OUTPUT_DIR", "outputs"))
VOICES_DIR = Path(os.environ.get("FLEXITALK_VOICES_DIR", "voices"))
MAX_AUDIO_OUTPUTS = 20

# チェックポイントパスをキャッシュ（毎回ダウンロードしない）
_CHECKPOINT_PATH: str | None = None

# 拡張子のドットを削除
EXTENSION_CHOICES = [ext.lstrip(".") for ext in VOICE_EXTENSIONS]


def _get_checkpoint_path() -> str:
    global _CHECKPOINT_PATH
    if _CHECKPOINT_PATH is None:
        _CHECKPOINT_PATH = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
        )
    return _CHECKPOINT_PATH


def _extract_non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def synthesize_text_lines(
    text_list: list[str],
    audio_outputs: list[str],
    text: str,
    voice_name: str,
    use_last_audio: bool,
    last_audio_path: str,
    num_steps: int = 20,
    text_cfg_scale: float = 3.0,
    speaker_cfg_scale: float = 5.0,
    output_format: str = "ogg",
):
    """テキストを行ごとに分割して合成し、生成された音声ファイルのパスをリストに追加して返すジェネレーター

    - 空行は無視
    - 最大行数は `MAX_AUDIO_OUTPUTS` に制限
    """
    audio_outputs = list(audio_outputs)
    lines = _extract_non_empty_lines(text)
    if not lines:
        raise gr.Error("テキストを入力してください。")
    if len(lines) > MAX_AUDIO_OUTPUTS:
        raise gr.Error(f"最大 {MAX_AUDIO_OUTPUTS} 行までです。")
    text_list = list(text_list) + lines[:MAX_AUDIO_OUTPUTS]
    yield [text_list, audio_outputs, last_audio_path]

    for i in range(min(len(lines), MAX_AUDIO_OUTPUTS)):
        if use_last_audio and last_audio_path:
            voice_name = last_audio_path
        audio = synthesize(
            lines[i],
            voice_name,
            num_steps,
            text_cfg_scale,
            speaker_cfg_scale,
            output_format,
        )
        audio_outputs.append(audio)
        last_audio_path = audio
        yield [text_list, audio_outputs, last_audio_path]


def _voice_list_dropdown() -> gr.Dropdown:
    return gr.Dropdown(
        label="リファレンスボイス名",
        choices=get_voice_file_choices(VOICES_DIR),
        value=NO_VOICE_OPTION,
    )


def synthesize(
    text: str,
    voice_name: str,
    num_steps: int = 20,
    text_cfg_scale: float = 3.0,
    speaker_cfg_scale: float = 5.0,
    output_format: str = "ogg",
) -> str:
    """テキストとボイス名を受け取り、指定された形式の音声ファイルのパスを返す。

    初回呼び出し時にモデルをダウンロード＆ロード。
    2回目以降はキャッシュされたモデルを即座に再利用。
    """
    if not text.strip():
        raise gr.Error("テキストを入力してください。")

    ckpt_path = _get_checkpoint_path()
    key = RuntimeKey(
        checkpoint=ckpt_path,
        model_device=MODEL_DEVICE,
        codec_device=CODEC_DEVICE,
        model_precision=MODEL_PRECISION,
        codec_precision=CODEC_PRECISION,
    )
    runtime, _ = get_cached_runtime(key)

    ref_wav = None
    if voice_name != NO_VOICE_OPTION:
        candidate_path = VOICES_DIR / voice_name
        if candidate_path.exists() and candidate_path.is_file():
            ref_wav = str(candidate_path)

    req = SamplingRequest(
        text=text.strip(),
        ref_wav=ref_wav,
        no_ref=voice_name == NO_VOICE_OPTION,
        num_steps=num_steps,
        cfg_scale_text=text_cfg_scale,
        cfg_scale_speaker=speaker_cfg_scale,
    )
    result = runtime.synthesize(req)

    audio_tensor: torch.Tensor = result.audio.cpu()
    sample_rate = int(result.sample_rate)

    # ボイス名に応じたサブディレクトリに保存
    out_subdir = OUT_DIR / _get_output_subdir(voice_name)
    out_subdir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    tmp_path = str(out_subdir / f"{timestamp}.{output_format}")
    torchaudio.save(tmp_path, audio_tensor, sample_rate)

    return tmp_path


# ── UI 構築 ───────────────────────────────────────────
with gr.Blocks(title="FlexiTalk") as demo:
    # 生成されたテキストのリスト
    text_list = gr.State([])
    # 生成された音声ファイルのパスのリスト
    audio_outputs = gr.State([])
    # 最後に合成した音声のパス
    last_audio_path = gr.State("")
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="テキスト入力",
                placeholder="ここに読み上げたい日本語テキストを入力してください…",
                lines=5,
            )
            voice_dropdown = _voice_list_dropdown()
            use_last_audio_checkbox = gr.Checkbox(
                label="最後に合成した音声をリファレンスにする",
                value=False,
            )
            refresh_voice_list_btn = gr.Button(
                "一覧更新",
                variant="secondary",
            )
            num_steps_slider = gr.Slider(
                label="ステップ数",
                minimum=1,
                maximum=100,
                value=20,
                step=1,
            )
            text_cfg_scale_slider = gr.Slider(
                label="Text CFG Scale",
                minimum=0.0,
                maximum=10.0,
                value=3.0,
                step=1/4,
            )
            speaker_cfg_scale_slider = gr.Slider(
                label="Speaker CFG Scale",
                minimum=0.0,
                maximum=10.0,
                value=3.0,
                step=1/4,
            )
            output_format_radio = gr.Radio(
                label="出力形式",
                choices=EXTENSION_CHOICES,
                value="ogg",
            )
            synthesize_btn = gr.Button(
                "🔊 合成",
                variant="primary",
                size="lg",
            )
            with gr.Accordion("情報", open=False):
                gr.Markdown(
                    f"- モデル: `{HF_REPO_ID}`\n"
                    f"- デバイス: `{MODEL_DEVICE}`\n"
                    f"- 精度: `{MODEL_PRECISION}`\n"
                )
        with gr.Column(scale=1):

            @gr.render(inputs=[text_list, audio_outputs])
            def render_audio_outputs(_text_list, audio_list) -> None:
                with gr.Column():
                    for i, text in enumerate(_text_list):
                        audio = audio_list[i] if i < len(audio_list) else None
                        if audio is None:
                            gr.Markdown(f"**合成中:** {text}")
                        else:
                            gr.Audio(
                                key=f"audio-{i}",
                                value=audio,
                                label=text,
                                interactive=False,
                            )

            erase_history_btn = gr.Button(
                "🗑️ 履歴を消去",
            )

    refresh_voice_list_btn.click(
        fn=_voice_list_dropdown,
        inputs=[],
        outputs=[voice_dropdown],
    )

    synthesize_btn.click(
        fn=synthesize_text_lines,
        inputs=[
            text_list,
            audio_outputs,
            text_input,
            voice_dropdown,
            use_last_audio_checkbox,
            last_audio_path,
            num_steps_slider,
            text_cfg_scale_slider,
            speaker_cfg_scale_slider,
            output_format_radio,
        ],
        outputs=[text_list, audio_outputs, last_audio_path],
    )

    # Enter キーでも合成実行
    text_input.submit(
        fn=synthesize_text_lines,
        inputs=[
            text_list,
            audio_outputs,
            text_input,
            voice_dropdown,
            use_last_audio_checkbox,
            last_audio_path,
            num_steps_slider,
            text_cfg_scale_slider,
            speaker_cfg_scale_slider,
            output_format_radio,
        ],
        outputs=[text_list, audio_outputs, last_audio_path],
    )

    # 履歴を消去ボタン
    erase_history_btn.click(
        fn=lambda: ([], [], ""),
        inputs=[],
        outputs=[text_list, audio_outputs, last_audio_path],
    )


def load_model():
    ckpt_path = _get_checkpoint_path()
    key = RuntimeKey(
        checkpoint=ckpt_path,
        model_device=MODEL_DEVICE,
        codec_device=CODEC_DEVICE,
        model_precision=MODEL_PRECISION,
        codec_precision=CODEC_PRECISION,
    )
    print(f"loading model {ckpt_path}...")
    runtime, _ = get_cached_runtime(key)
    print("model loaded")


def main() -> None:
    # UI起動前にバックグラウンド開始
    threading.Thread(
        target=load_model,
        daemon=True
    ).start()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[str(OUT_DIR), str(VOICES_DIR)],
    )


if __name__ == "__main__":
    main()
