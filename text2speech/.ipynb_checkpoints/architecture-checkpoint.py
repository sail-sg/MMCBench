import torch
import torch.nn as nn
from transformers import VitsModel, AutoTokenizer
from utils import resample_speech

class SpeechSynthesizer(nn.Module):
    """
    Generates speech from text using a specified TTS model.

    :param model_name: Name of the TTS model to be used for speech synthesis.
    :param device: The device to run the model on ('cuda' for GPU or 'cpu').
    """
    def __init__(self, model_name, device='cuda'):
        super(SpeechSynthesizer, self).__init__()
        self.model_name = model_name
        self.device = device
        self.sr = None  # Initialize the sample rate

        if 'mms-tts-eng' == model_name:
            self.model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            self.sr = self.model.config.sampling_rate
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @torch.no_grad()
    def forward(self, inputs):
        """
        Generates speech from a list of text inputs.

        :param inputs: List of text inputs for speech synthesis.
        :return: List of synthesized speech signals.
        """
        if 'mms-tts-eng' == self.model_name:
            outputs = []
            ## advoid out-of-memory
            for input in inputs:
                input = self.tokenizer(input, return_tensors="pt").to('cuda')
                speech = self.model(**input).waveform
                outputs.append(speech.squeeze().cpu().numpy())
        else:
            raise ValueError(f"Model '{self.model_name}' not supported in forward method.")

        # Resample the output speech to 16 kHz
        # Sometimes, the outputs may be some paths for the speech, and it still works here
        outputs = resample_speech(outputs, self.sr)
        return outputs
