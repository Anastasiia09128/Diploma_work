import re
import textwrap
from typing import List


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int((secs - int(secs)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(secs):02},{millis:03}"


def clean_text_for_subs(text: str) -> str:
   
    text = re.sub(r"\b(\w+)([\s,]+\1\b)+", r"\1", text, flags=re.IGNORECASE)

    fillers = r"\b(er|eh|hmm|mm-hmm|uh-huh|uh-uh|oh|ah|uh|huh|erm|um)\b[,\s]*"
    text = re.sub(fillers, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def split_text_into_lines(text: str, max_line_len: int = 35) -> List[str]:
    
    return textwrap.wrap(text, width=max_line_len)


def split_segment_text_into_blocks(text: str, max_chars_block: int = 70) -> List[str]:
    
    text = clean_text_for_subs(text)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    blocks: List[str] = []

    for sent in sentences:
        content_only = re.sub(r"[\W_]+", "", sent, flags=re.UNICODE)
        if not content_only:
            continue

        words = sent.split()
        current_words: List[str] = []

        for w in words:
            test_block = " ".join(current_words + [w])
            if len(test_block) > max_chars_block:
                if current_words:
                    blocks.append(" ".join(current_words))
                    current_words = [w]
                else:
                    current_words = [w]
            else:
                current_words.append(w)

        if current_words:
            blocks.append(" ".join(current_words))

    return blocks


def segments_to_srt(
    segments,
    output_path: str,
    max_chars_block: int = 70,
    max_line_len: int = 35,
    min_block_dur: float = 1.0,
) -> str:
   
    srt_lines = []
    index = 1

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"].strip()

        if not seg_text:
            continue

        blocks = split_segment_text_into_blocks(
            seg_text, max_chars_block=max_chars_block
        )
        if not blocks:
            continue

        seg_duration = seg_end - seg_start
        total_chars = sum(len(b) for b in blocks)
        if total_chars == 0 or seg_duration <= 0:
            continue

        n_blocks = len(blocks)

        base_durs = [(len(b) / total_chars) * seg_duration for b in blocks]

        if seg_duration >= n_blocks * min_block_dur:
            extra_time = seg_duration - n_blocks * min_block_dur
            base_sum = sum(base_durs) or 1.0

            block_durs = [
                min_block_dur + extra_time * (bd / base_sum) for bd in base_durs
            ]
        else:
            block_durs = base_durs

        dur_sum = sum(block_durs)
        if dur_sum > 0:
            scale = seg_duration / dur_sum
            block_durs = [d * scale for d in block_durs]

        current_time = seg_start

        for block_text, block_dur in zip(blocks, block_durs):
            block_end = current_time + block_dur

            start_ts = format_timestamp(current_time)
            end_ts = format_timestamp(block_end)

            lines = split_text_into_lines(block_text, max_line_len=max_line_len)

            srt_lines.append(str(index))
            srt_lines.append(f"{start_ts} --> {end_ts}")
            srt_lines.extend(lines)
            srt_lines.append("")

            index += 1
            current_time = block_end

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print(f"SRT-файл сохранён как: {output_path}")
    return output_path
