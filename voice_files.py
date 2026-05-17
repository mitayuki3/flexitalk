from pathlib import Path

NO_VOICE_OPTION = "無指定"
VOICE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def get_voice_file_choices(voices_dir: Path) -> list[str]:
    if not voices_dir.exists():
        return [NO_VOICE_OPTION]

    voice_files = [
        str(path.relative_to(voices_dir)).replace("\\", "/")
        for path in voices_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VOICE_EXTENSIONS
    ]
    voice_files.sort()
    return [NO_VOICE_OPTION] + voice_files


def _get_output_subdir(voice_name: str) -> Path:
    if voice_name == NO_VOICE_OPTION:
        return Path("no_voice")
    return Path(voice_name).with_suffix("")
