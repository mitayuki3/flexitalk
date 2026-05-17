"""
Irodori-TTS ベースモデルを使用した Gradio GUI
"""

from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    get_cached_runtime,
    default_runtime_device,
)
from voice_files import (
    NO_VOICE_OPTION,
    VOICES_DIR,
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

OUT_DIR = Path("outputs")

# チェックポイントパスをキャッシュ（毎回ダウンロードしない）
_CHECKPOINT_PATH: str | None = None


def _get_checkpoint_path() -> str:
    global _CHECKPOINT_PATH
    if _CHECKPOINT_PATH is None:
        _CHECKPOINT_PATH = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
        )
    return _CHECKPOINT_PATH


def _voice_list_dropdown() -> gr.Dropdown:
    return gr.Dropdown(
        label="TTSボイス名",
        choices=get_voice_file_choices(),
        value=NO_VOICE_OPTION,
    )


def synthesize(text: str, voice_name: str, num_steps: int = 20) -> str:
    """テキストとボイス名を受け取り、MP3 ファイルのパスを返す。

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
    )
    result = runtime.synthesize(req)

    # torch.Tensor → numpy → pydub → MP3 に変換
    audio_np = result.audio.cpu().numpy()
    sample_rate = int(result.sample_rate)

    # float32 numpy → pydub AudioSegment
    audio_int16 = (audio_np * 32767).astype(np.int16)
    segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16bit
        channels=1,
    )

    # ボイス名に応じたサブディレクトリに MP3 を保存
    out_subdir = OUT_DIR / _get_output_subdir(voice_name)
    out_subdir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    tmp_path = str(out_subdir / f"{timestamp}.mp3")
    segment.export(tmp_path, format="mp3")

    return tmp_path


# ── UI 構築 ───────────────────────────────────────────
with gr.Blocks(title="FlexiTalk") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="テキスト入力",
                placeholder="ここに読み上げたい日本語テキストを入力してください…",
                lines=5,
            )
            voice_dropdown = _voice_list_dropdown()
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
            synthesize_btn = gr.Button(
                "🔊 合成",
                variant="primary",
                size="lg",
            )
        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="音声出力",
                type="filepath",
                interactive=False,
            )
            with gr.Accordion("情報", open=False):
                gr.Markdown(
                    f"- モデル: `{HF_REPO_ID}`\n"
                    f"- デバイス: `{MODEL_DEVICE}`\n"
                    f"- 精度: `{MODEL_PRECISION}`\n"
                    f"- 出力形式: MP3\n"
                )

    refresh_voice_list_btn.click(
        fn=_voice_list_dropdown,
        inputs=[],
        outputs=[voice_dropdown],
    )

    synthesize_btn.click(
        fn=synthesize,
        inputs=[text_input, voice_dropdown, num_steps_slider],
        outputs=audio_output,
    )

    # Enter キーでも合成実行
    text_input.submit(
        fn=synthesize,
        inputs=[text_input, voice_dropdown, num_steps_slider],
        outputs=audio_output,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
