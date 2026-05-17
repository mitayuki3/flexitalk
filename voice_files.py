from pathlib import Path

NO_VOICE_OPTION = "無指定"
VOICE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
VOICES_DIR = Path("voices")


def get_voice_file_choices() -> list[str]:
    if not VOICES_DIR.exists():
        return [NO_VOICE_OPTION]

    voice_files = [
        str(path.relative_to(VOICES_DIR)).replace("\\", "/")
        for path in VOICES_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VOICE_EXTENSIONS
    ]
    voice_files.sort()
    return [NO_VOICE_OPTION] + voice_files


def _get_output_subdir(voice_name: str) -> Path:
    if voice_name == NO_VOICE_OPTION:
        return Path("no_voice")
    return Path(voice_name).with_suffix("")
