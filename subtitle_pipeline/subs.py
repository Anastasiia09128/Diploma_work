import re
import textwrap
from typing import List

import spacy


MAX_LINE_LEN = 39
MAX_LINES = 2
MAX_CHARS_BLOCK = MAX_LINE_LEN * MAX_LINES
MAX_READING_SPEED = 17
MIN_GAP_BETWEEN_SUBS = 1.0

nlp = spacy.load("en_core_web_sm")


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


def split_text_into_lines(text: str, max_line_len: int = MAX_LINE_LEN) -> List[str]:
    return textwrap.wrap(text, width=max_line_len)


def is_valid_block_text(
    text: str,
    max_chars_block: int = MAX_CHARS_BLOCK,
    max_line_len: int = MAX_LINE_LEN,
    max_lines: int = MAX_LINES,
) -> bool:
    lines = split_text_into_lines(text, max_line_len=max_line_len)

    return (
        len(text) <= max_chars_block
        and len(lines) <= max_lines
        and all(len(line) <= max_line_len for line in lines)
    )


def split_segment_text_into_blocks(
    text: str,
    max_chars_block: int = MAX_CHARS_BLOCK,
    max_line_len: int = MAX_LINE_LEN,
    max_lines: int = MAX_LINES,
) -> List[str]:
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

            if not is_valid_block_text(
                test_block,
                max_chars_block=max_chars_block,
                max_line_len=max_line_len,
                max_lines=max_lines,
            ):
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


def reading_speed(block) -> float:
    duration = block["end"] - block["start"]
    if duration <= 0:
        return float("inf")
    return len(block["text"]) / duration


def refresh_block(block):
    block = block.copy()
    block["text"] = clean_text_for_subs(block["text"])
    block["lines"] = split_text_into_lines(block["text"])
    return block


def block_is_valid(block) -> bool:
    return is_valid_block_text(block["text"])


def adjust_timings_for_reading_speed(
    blocks,
    max_reading_speed: float = MAX_READING_SPEED,
    min_gap: float = MIN_GAP_BETWEEN_SUBS,
):
    if not blocks:
        return blocks

    adjusted = [b.copy() for b in blocks]

    for i, block in enumerate(adjusted):
        text_len = len(block["text"])
        current_start = block["start"]
        current_end = block["end"]
        current_duration = current_end - current_start

        if current_duration <= 0:
            continue

        current_speed = text_len / current_duration

        if current_speed <= max_reading_speed:
            continue

        needed_duration = text_len / max_reading_speed
        desired_end = current_start + needed_duration

        if i < len(adjusted) - 1:
            next_start = adjusted[i + 1]["start"]
            max_possible_end = next_start - min_gap
        else:
            max_possible_end = desired_end

        if max_possible_end > current_end:
            new_end = min(desired_end, max_possible_end)

            if new_end > current_end:
                block["end"] = new_end
                block["timing_adjusted"] = True
            else:
                block["timing_adjusted"] = False
        else:
            block["timing_adjusted"] = False

    return adjusted


def segments_to_blocks(
    segments,
    max_chars_block: int = MAX_CHARS_BLOCK,
    max_line_len: int = MAX_LINE_LEN,
    max_lines: int = MAX_LINES,
    min_block_dur: float = 1.0,
    adjust_reading_time: bool = True,
):
    subtitle_blocks = []

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"].strip()

        if not seg_text:
            continue

        blocks = split_segment_text_into_blocks(
            seg_text,
            max_chars_block=max_chars_block,
            max_line_len=max_line_len,
            max_lines=max_lines,
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
                min_block_dur + extra_time * (bd / base_sum)
                for bd in base_durs
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

            subtitle_blocks.append(
                {
                    "start": current_time,
                    "end": block_end,
                    "original_start": current_time,
                    "original_end": block_end,
                    "text": block_text,
                    "lines": split_text_into_lines(
                        block_text,
                        max_line_len=max_line_len,
                    ),
                    "timing_adjusted": False,
                }
            )

            current_time = block_end

    if adjust_reading_time:
        subtitle_blocks = adjust_timings_for_reading_speed(
            subtitle_blocks,
            max_reading_speed=MAX_READING_SPEED,
            min_gap=MIN_GAP_BETWEEN_SUBS,
        )

    return subtitle_blocks


def is_bad_boundary(left_text, right_text) -> bool:
    left_doc = nlp(left_text)
    right_doc = nlp(right_text)

    if len(left_doc) == 0 or len(right_doc) == 0:
        return False

    left_last = left_doc[-1]
    right_first = right_doc[0]

    left_word = left_last.text.lower()
    right_pos = right_first.pos_

    if left_last.pos_ == "ADP" and right_pos in {"NOUN", "PROPN", "PRON", "DET", "ADJ"}:
        return True

    if left_last.pos_ == "DET" and right_pos in {"NOUN", "PROPN", "ADJ"}:
        return True

    if left_word == "to" and right_pos == "VERB":
        return True

    if left_word in {"not", "n't"} and right_pos in {"VERB", "AUX", "ADJ"}:
        return True

    if left_last.pos_ == "AUX" and right_pos in {"VERB", "AUX"}:
        return True

    if left_last.pos_ == "ADJ" and right_pos in {"NOUN", "PROPN"}:
        return True

    if left_last.pos_ == "PROPN" and right_pos == "PROPN":
        return True

    if left_last.pos_ == "PRON" and right_pos in {"VERB", "AUX"}:
        return True

    return False


def syntax_boundary_refinement(blocks):
    if len(blocks) <= 1:
        return blocks

    refined = []
    i = 0

    while i < len(blocks):
        current = refresh_block(blocks[i])

        if i < len(blocks) - 1:
            next_block = refresh_block(blocks[i + 1])

            if is_bad_boundary(current["text"], next_block["text"]):
                candidate = {
                    "start": current["start"],
                    "end": next_block["end"],
                    "original_start": current.get("original_start", current["start"]),
                    "original_end": next_block.get("original_end", next_block["end"]),
                    "text": current["text"] + " " + next_block["text"],
                    "timing_adjusted": (
                        current.get("timing_adjusted", False)
                        or next_block.get("timing_adjusted", False)
                    ),
                }

                candidate = refresh_block(candidate)

                if block_is_valid(candidate) and reading_speed(candidate) <= MAX_READING_SPEED:
                    refined.append(candidate)
                    i += 2
                    continue

        refined.append(current)
        i += 1

    return refined


def build_improved_en_blocks(
    segments,
    max_chars_block: int = MAX_CHARS_BLOCK,
    max_line_len: int = MAX_LINE_LEN,
    max_lines: int = MAX_LINES,
    min_block_dur: float = 1.0,
):
    blocks = segments_to_blocks(
        segments,
        max_chars_block=max_chars_block,
        max_line_len=max_line_len,
        max_lines=max_lines,
        min_block_dur=min_block_dur,
        adjust_reading_time=True,
    )

    blocks = syntax_boundary_refinement(blocks)

    blocks = adjust_timings_for_reading_speed(
        blocks,
        max_reading_speed=MAX_READING_SPEED,
        min_gap=MIN_GAP_BETWEEN_SUBS,
    )

    return blocks


def segments_to_srt(
    segments,
    output_path: str,
    max_chars_block: int = MAX_CHARS_BLOCK,
    max_line_len: int = MAX_LINE_LEN,
    min_block_dur: float = 1.0,
) -> str:
    srt_lines = []
    index = 1

    blocks = build_improved_en_blocks(
        segments,
        max_chars_block=max_chars_block,
        max_line_len=max_line_len,
        max_lines=MAX_LINES,
        min_block_dur=min_block_dur,
    )

    for block in blocks:
        start_ts = format_timestamp(block["start"])
        end_ts = format_timestamp(block["end"])

        lines = split_text_into_lines(
            block["text"],
            max_line_len=max_line_len,
        )

        if not lines:
            continue

        srt_lines.append(str(index))
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.extend(lines)
        srt_lines.append("")

        index += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print(f"SRT-файл сохранён как: {output_path}")
    return output_path
