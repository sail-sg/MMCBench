import torch
import torch.nn as nn
import numpy as np
import librosa
from sentence_transformers import SentenceTransformer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def resample_speech(outputs, original_sr, target_sr=16000):
    """
    Resamples speech signals to a fixed sample rate.
    
    :param outputs: A list of speech signals (numpy arrays) or file paths.
    :param original_sr: Original sample rate of the speech signals.
    :param target_sr: Target sample rate for resampling. Default is 16000 Hz.
    :return: List of resampled speech signals.
    """
    resampled_speech = []
    for item in outputs:
        # Check if the item is a file path
        if isinstance(item, str):
            speech, _ = librosa.load(item, sr=target_sr)
        elif isinstance(item, np.ndarray):
            speech = item
            # Resample only if the original sample rate is different from the target
            if original_sr != target_sr:
                speech = librosa.resample(speech, orig_sr=original_sr, target_sr=target_sr)
        else:
            raise ValueError("Each item must be a speech array or a file path.")
        resampled_speech.append(speech)
    return resampled_speech

class SentenceSimilarityCalculator(nn.Module):
    """
    Calculates the cosine similarity between two sets of sentences using a specified NLP model.

    :param model_name: Name of the NLP model to be used for encoding sentences.
    :param device: The device to run the model on ('cuda' for GPU or 'cpu').
    """
    def __init__(self, model_name='sentence-transformers/sentence-t5-large', device='cuda'):
        super(SentenceSimilarityCalculator, self).__init__()
        self.text_encoder = SentenceTransformer(model_name).to(device).eval()

    @torch.no_grad()
    def forward(self, sentences_1, sentences_2):
        """
        Computes the cosine similarity between two sets of sentences.

        :param sentences_1: The first set of sentences to compare.
        :param sentences_2: The second set of sentences to compare.
        :return: A tensor containing cosine similarity scores between each pair of sentences.
        """
        # Ensure both inputs are in list format
        if isinstance(sentences_1, str):
            sentences_1 = [sentences_1]
        if isinstance(sentences_2, str):
            sentences_2 = [sentences_2]

        # Encode the sentences to embeddings
        encoding_1 = self.text_encoder.encode(sentences_1, convert_to_tensor=True)
        encoding_2 = self.text_encoder.encode(sentences_2, convert_to_tensor=True)

        # Compute cosine similarity
        cos = nn.CosineSimilarity(dim=1)
        similarity_scores = cos(encoding_1, encoding_2)
        return similarity_scores.cpu().numpy()

class SpeechContentSimilarity(nn.Module):
    """
    Computes the similarity between the content of two speech inputs based on their transcriptions.

    :param model_name: Name of the speech-to-text model to be used for transcription.
    :param device: The device to run the model on ('cuda' for GPU or 'cpu').
    """
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device='cuda'):
        super(SpeechContentSimilarity, self).__init__()
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.sentence_similarity_calculator = SentenceSimilarityCalculator(device=device) 

    @torch.no_grad()
    def forward(self, speech_1, speech_2):
        """
        Computes the similarity between two speech inputs.

        :param speech_1: The first speech input (as an audio file path or numpy array).
        :param speech_2: The second speech input (as an audio file path or numpy array).
        :return: Cosine similarity score between the content of the two speech inputs.
        """
        transcription_1 = self.transcribe_speech(speech_1)
        transcription_2 = self.transcribe_speech(speech_2)
        return self.sentence_similarity_calculator(transcription_1, transcription_2)

    def transcribe_speech(self, inputs):
        """
        Transcribes speech input to text using a speech-to-text model.

        :param inputs: Speech input to transcribe (as an audio file path or numpy array).
        :return: Transcription of the speech input as a string.
        """
        input_features = self.processor(inputs, sampling_rate=16000, padding='longest', return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(input_features).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return [transcription.strip().lower() for transcription in transcriptions]