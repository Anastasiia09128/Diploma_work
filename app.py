import gradio as gr
from subtitle_pipeline import process_video


def run_pipeline(video_file, model_size):
    if not video_file:
        raise gr.Error("Пожалуйста, загрузите видео.")

    # Gradio может передать либо строку (путь к файлу), либо объект.
    if isinstance(video_file, str):
        video_path = video_file
    else:
        # на всякий случай пытаемся взять name/path
        video_path = getattr(video_file, "name", None) or getattr(video_file, "path", None)
        if video_path is None:
            raise gr.Error("Не удалось определить путь к видеофайлу.")

    print("Запускаем пайплайн для:", video_path)
    en_srt_path, ru_srt_path = process_video(video_path, model_size=model_size)

    # Gradio File ждёт строку пути
    return str(en_srt_path), str(ru_srt_path)



with gr.Blocks(title="MoviSub — генератор субтитров") as demo:
    gr.Markdown(
        """
        # 🎬 MoviSub

        Загрузите видео, и сервис:
        1. Выделит аудио  
        2. Распознает речь (Whisper)  
        3. Создаст английские субтитры (`.srt`)  
        4. Переведёт субтитры на русский и сохранит второй `.srt`  

        Затем вы сможете скачать оба файла.
        """
    )

    with gr.Row():
        video_input = gr.Video(label="Видео файл", sources=["upload"])
        model_size = gr.Dropdown(
            ["tiny", "base", "small", "medium"],
            value="small",
            label="Размер модели Whisper",
            info="Чем больше модель, тем лучше качество, но дольше обработка.",
        )

    run_btn = gr.Button("🚀 Обработать видео")

    with gr.Row():
        en_srt_output = gr.File(label="Английские субтитры (.srt)")
        ru_srt_output = gr.File(label="Русские субтитры (.srt)")

    run_btn.click(
        fn=run_pipeline,
        inputs=[video_input, model_size],
        outputs=[en_srt_output, ru_srt_output],
    )

if __name__ == "__main__":
    demo.launch()
