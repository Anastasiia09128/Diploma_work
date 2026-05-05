from pathlib import Path

from .audio import extract_audio
from .stt import WhisperASR
from .subs import segments_to_srt
from .translate import build_ru_segments, segments_to_srt_ru


def make_srt_filename(video_path: str, prefix: str) -> Path:
    """
    Делает имя файла вида:
    <outputs>/<prefix>_<basename>.srt
    """
    video_path = Path(video_path)
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    base = video_path.stem  
    return outputs_dir / f"{prefix}_{base}.srt"


def process_video(video_path: str, model_size: str = "small") -> tuple[str, str]:
    
    video_path = str(video_path)
    print(f"Обработка видео: {video_path}")

    wav_path = extract_audio(video_path)
    print(f"Аудио сохранено в файл: {wav_path}")

    asr = WhisperASR(model_size=model_size)
    raw_text, segments = asr.transcribe(str(wav_path))

    print("\nПервые 1000 символов распознанного текста:")
    print(raw_text[:1000])

    en_srt_path = make_srt_filename(video_path, prefix="en_subs")
    en_srt_path_str = segments_to_srt(
        segments,
        output_path=str(en_srt_path),
        max_chars_block=78,
        max_line_len=39,
        min_block_dur=1.0,
    )

    segments_ru = build_ru_segments(segments)

    ru_srt_path = make_srt_filename(video_path, prefix="ru_subs")
    ru_srt_path_str = segments_to_srt_ru(
        segments_ru,
        output_path=str(ru_srt_path),
        max_chars_block=78,
        max_line_len=39,
        min_block_dur=1.0,
    )

    return en_srt_path_str, ru_srt_path_str
