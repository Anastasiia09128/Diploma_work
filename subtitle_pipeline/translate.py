import re
import textwrap
from typing import List, Dict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .subs import clean_text_for_subs, format_timestamp

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Устройство для перевода:", device)

MODEL_DIR = Path("/mnt/d/opus-mt-en-ru")

print(f"Загрузка модели перевода из локальной папки: {MODEL_DIR}")


tokenizer_mt = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

model_mt = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
).to(device)



def translate_batch_en_ru(texts: List[str], max_new_tokens: int = 128) -> List[str]:
    
    if not texts:
        return []

    clean_texts = [t if t.strip() else "." for t in texts]

    batch = tokenizer_mt(
        clean_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        generated = model_mt.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            no_repeat_ngram_size=3,
        )

    decoded = [
        tokenizer_mt.decode(g, skip_special_tokens=True) for g in generated
    ]
    decoded = [d.strip() for d in decoded]
    return decoded


def build_ru_segments(segments) -> List[Dict]:
    
    texts_en_seg = [clean_text_for_subs(seg["text"]) for seg in segments]
    texts_ru_seg = translate_batch_en_ru(texts_en_seg)

    segments_ru: List[Dict] = []
    for seg, text_ru in zip(segments, texts_ru_seg):
        segments_ru.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": text_ru,
            }
        )
    return segments_ru




def clean_text_for_subs_ru(text: str) -> str:
    
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    return text


def split_text_into_lines_ru(text: str, max_line_len: int = 39) -> List[str]:
    
    return textwrap.wrap(text, width=max_line_len)


def split_segment_text_into_blocks_ru(
    text: str, max_chars_block: int = 70
) -> List[str]:
    
    text = clean_text_for_subs_ru(text)

    sentences = re.split(r"(?<=[.!?…])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    PREPS = {
        "в",
        "во",
        "с",
        "со",
        "к",
        "ко",
        "у",
        "о",
        "об",
        "обо",
        "на",
        "за",
        "по",
        "от",
        "до",
        "из",
        "изо",
        "без",
        "для",
        "под",
        "подо",
        "над",
        "при",
        "про",
    }

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
                    last_word = current_words[-1].lower()
                    if last_word in PREPS:
                        prep = current_words.pop()
                        if current_words:
                            blocks.append(" ".join(current_words))
                        current_words = [prep, w]
                    else:
                        blocks.append(" ".join(current_words))
                        current_words = [w]
                else:
                    current_words = [w]
            else:
                current_words.append(w)

        if current_words:
            blocks.append(" ".join(current_words))

    return blocks


def segments_to_srt_ru(
    segments,
    output_path: str,
    max_chars_block: int = 70,
    max_line_len: int = 39,
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

        blocks = split_segment_text_into_blocks_ru(
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

            start_ts = format_timestamp(current_time)
            end_ts = format_timestamp(block_end)

            lines = split_text_into_lines_ru(
                block_text, max_line_len=max_line_len
            )
            if not lines:
                current_time = block_end
                continue

            srt_lines.append(str(index))
            srt_lines.append(f"{start_ts} --> {end_ts}")
            srt_lines.extend(lines)
            srt_lines.append("")

            index += 1
            current_time = block_end

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print(f"RU SRT-файл сохранён как: {output_path}")
    return output_path
