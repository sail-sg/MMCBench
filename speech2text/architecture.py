import torch
import torch.nn as nn
from transformers import WhisperTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC

class TranscriptionGenerator(nn.Module):
    """
    A class for generating transcriptions from audio data using various models.
    """
    def __init__(self, model_name, device='cuda'):
        """
        Initializes the TranscriptionGenerator.

        :param model_name: Name of the model to be used for transcription.
        :param device: The device to run the model on (default is 'cuda').
        """
        super(TranscriptionGenerator, self).__init__()
        self.model_name = model_name
        self.device = device
        ## should maintain a consistent style for transcription (no matter what models used here) ##
        self._normalize = WhisperTokenizer.from_pretrained("openai/whisper-base")._normalize
        
        # Initialize model based on the provided model name
        if 'wav2vec2' in model_name:
            self.processor = Wav2Vec2Processor.from_pretrained(f"facebook/{model_name}")
            self.model = Wav2Vec2ForCTC.from_pretrained(f"facebook/{model_name}").to(device)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @torch.no_grad()
    def forward(self, inputs, sample_rate=16000):
        """
        Processes the input audio and generates transcriptions.

        :param inputs: Input audio data.
        :param sample_rate: Sample rate of the input audio (default is 16000Hz).
        :return: List of transcriptions.
        """
        if 'wav2vec2' in self.model_name:
            input_features = self.processor(inputs, sampling_rate=sample_rate, padding='longest', return_tensors="pt").input_values.to(self.device)
            logits = self.model(input_features).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        else:
            raise ValueError(f"Model '{self.model_name}' not supported in forward method.")
            
        # Normalize the transcriptions
        transcriptions = [self._normalize(transcription.strip()) for transcription in transcriptions]
        return transcriptions
