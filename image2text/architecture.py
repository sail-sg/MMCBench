import os
import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

class CaptionGenerator(nn.Module):
    def __init__(self, model_name, device='cuda'):
        """
        Initializes the CaptionGenerator with a specific model.

        :param model_name: Name of the model to be used for caption generation.
        :param device: The device to run the model on. Defaults to 'cuda'.
        """
        super(CaptionGenerator, self).__init__()
        self.model_name = model_name
        self.device = device

        if model_name == 'blip_base':
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to(device)
        elif 'instructblip' in model_name:
            self.processor = InstructBlipProcessor.from_pretrained(f"Salesforce/{model_name}")
            self.model = InstructBlipForConditionalGeneration.from_pretrained(f"Salesforce/{model_name}", torch_dtype=torch.float16).to(device)
        else:
            raise ValueError("Invalid model name provided. Please use 'blip_base' or a model name containing 'instructblip'.")

    @torch.no_grad()
    def forward(self, imgs, num_beams=1, max_new_tokens=256, instruction="Describe this image as detailed as possible."):
        """
        Generates captions for a list of images.

        :param imgs: List of images for caption generation.
        :param num_beams: Number of beams for beam search. Defaults to 1.
        :param max_new_tokens: Maximum new tokens for generation. Defaults to 256.
        :param instruction: Instruction text for captioning. Defaults to a generic description instruction.
        :return: List of generated captions.
        """
        if self.model_name == 'blip_base':
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device, torch.float16)
            outputs = self.model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)
        elif 'instructblip' in self.model_name:
            prompt = [instruction] * len(imgs)
            inputs = self.processor(images=imgs, text=prompt, return_tensors="pt").to(self.device, torch.float16)
            outputs = self.model.generate(**inputs, do_sample=False, min_length=1, repetition_penalty=1.5, length_penalty=1.0, num_beams=num_beams, max_length=max_new_tokens)
        else:
            raise NotImplementedError("Model processing not implemented for the provided model name.")

        captions = [catpion.strip() for catpion in self.processor.batch_decode(outputs, skip_special_tokens=True)]
        return captions