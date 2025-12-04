import gradio as gr
from subtitle_pipeline import process_video


def run_pipeline(video_file, model_size):
    """
    Функция-обёртка для Gradio.
    video_file — это объект с путём к временному файлу.
    """
    if video_file is None:
        return None, None

    # Gradio передаёт объект с атрибутом name (путь к файлу)
    video_path = video_file.name

    en_srt_path, ru_srt_path = process_video(video_path, model_size=model_size)

    # Gradio ожидает пути к файлам для загрузки
    return en_srt_path, ru_srt_path


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
