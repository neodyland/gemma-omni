from transformers import MimiModel, AutoFeatureExtractor
import torch
from torchaudio.transforms import Resample


class Mimi:
    model: MimiModel
    feature_extractor: AutoFeatureExtractor

    def __init__(self):
        self.model = MimiModel.from_pretrained("kyutai/mimi")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    def encode(self, audio: torch.Tensor, sr: int):
        audio = Resample(sr, self.feature_extractor.sampling_rate)(audio)
        inputs = self.feature_extractor(
            raw_audio=audio,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        return self.model.encode(inputs["input_values"]).audio_codes

    def decode(self, audio_codes: torch.Tensor):
        return self.model.decode(audio_codes)[0]


if __name__ == "__main__":
    import librosa

    mimi = Mimi()
    print(
        mimi.encode(
            torch.from_numpy(librosa.load("./data/sample.wav", sr=24000)[0]),
            24000,
        ).shape
    )
