import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

def bytes_to_speech(byte_data, dtype=np.float64):  # Using float64 if that was the original dtype
    speech_array = np.frombuffer(byte_data, dtype=dtype)
    return speech_array

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