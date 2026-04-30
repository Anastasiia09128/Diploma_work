import whisper


class WhisperASR:
    
    def __init__(self, model_size: str = "small"):
        self.device = "cpu"
        print(f"Инициализация Whisper ({model_size})... Устройство: {self.device}")

        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_path: str):
        
        print("Запуск распознавания...")

        result = self.model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            fp16=False,
        )

        raw_text = result["text"].strip()
        segments = result["segments"]

        return raw_text, segments
