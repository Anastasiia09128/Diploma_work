import gradio as gr
from pathlib import Path
import subprocess
import os

from subtitle_pipeline import process_video

def clear_outputs():
    out_dir = Path("outputs")
    if not out_dir.exists():
        return
    for p in out_dir.iterdir():
        if p.is_file():
            try:
                p.unlink()
            except OSError as e:
                print(f"Не удалось удалить {p}: {e}")


def burn_subtitles(video_path: str, srt_path: str, suffix: str) -> str:

    input_video = Path(video_path)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_video = output_dir / f"{input_video.stem}_{suffix}_subs.mp4"

    srt_path = Path(srt_path).resolve()
    input_video = input_video.resolve()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        f"subtitles='{srt_path.as_posix()}'",
        "-c:a",
        "copy",
        str(output_video),
    ]

    print("Видео с субтитрами командой:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Видео с субтитрами сохранено как:", output_video)

    return str(output_video)


def run_pipeline(video_file):
    if not video_file:
        raise gr.Error("Пожалуйста, загрузите видео.")
    
    clear_outputs() 



    if isinstance(video_file, str):
        video_path = video_file
    else:
        video_path = getattr(video_file, "name", None) or getattr(
            video_file, "path", None
        )
        if video_path is None:
            raise gr.Error("Не удалось определить путь к видеофайлу.")

    print("Запуск пайплайна для:", video_path)

    en_srt_path, ru_srt_path = process_video(video_path, model_size="small")

    video_with_en_subs = burn_subtitles(video_path, en_srt_path, suffix="en")
    video_with_ru_subs = burn_subtitles(video_path, ru_srt_path, suffix="ru")

    return video_with_en_subs, video_with_ru_subs, str(en_srt_path), str(ru_srt_path)


with gr.Blocks(
    title="Субтитры",
) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h1>🎬 Субтитры</h1>
            <p style="font-size: 1.05rem;">
                Прототип системы автоматического создания английских и русских субтитров к видео.
            </p>
        </div> 
        """
    )

    gr.Markdown("Загрузите видео")
    video_input = gr.File(
        label="Видео файл",
        file_types=["video"],
    )

    run_btn = gr.Button(
        "Создать субтитры",
        variant="primary",
    )

    gr.Markdown("Просмотр результата")

    with gr.Row():
        video_with_en_subs = gr.Video(
            label="Видео с оригинальными (EN) субтитрами",
            interactive=False,
        )
        video_with_ru_subs = gr.Video(
            label="Видео с русскими субтитрами",
            interactive=False,
        )

    gr.Markdown("Скачать субтитры")

    with gr.Row():
        en_srt_output = gr.File(
            label="Английские субтитры (.srt)",
        )
        ru_srt_output = gr.File(
            label="Русские субтитры (.srt)",
        )


    run_btn.click(
        fn=run_pipeline,
        inputs=[video_input],
        outputs=[video_with_en_subs, video_with_ru_subs, en_srt_output, ru_srt_output],
    )

if __name__ == "__main__":
    demo.launch(theme="Nymbo/Nymbo_Theme")
