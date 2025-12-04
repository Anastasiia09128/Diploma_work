import subprocess
from pathlib import Path


def extract_audio(input_video: str, sample_rate: int = 16000) -> Path:
   
    input_path = Path(input_video)
    stem = input_path.stem
    output_path = input_path.with_name(f"{stem}_audio.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        str(output_path),
    ]

    subprocess.run(cmd, check=True)
    return output_path
