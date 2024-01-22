import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline

class ImageGenerator(nn.Module):
    def __init__(self, model_name, device = 'cuda'):
        """
        Initializes the ImageGenerator with the specified model.

        :param model_name: Name of the model to be used for image generation.
        :param device: The device to run the model on ('cuda' for GPU or 'cpu').
        """
        super(ImageGenerator, self).__init__()
        self.model_name = model_name
        if model_name == 'stable-diffusion-v1-5':
            # Initialize the Stable Diffusion Pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
            self.pipe = self.pipe.to(device)
            # Disable the safety checker for the pipeline
            self.pipe.run_safety_checker = lambda images, device, dtype: (images, [0] * len(images))
            self.pipe.safety_checker = lambda clip_input, images: (images, [0] * len(images))
        else:
            # Raise an error for unsupported models
            raise ValueError(f"Model '{model_name}' not supported.")

    def forward(self, captions):
        """
        Generates images based on the provided captions.

        :param captions: A list of captions to generate images from.
        :return: A list of generated images.
        """
        if self.model_name == 'stable-diffusion-v1-5':
            images = self.pipe(captions).images
        else:
            raise ValueError(f"Model '{self.model_name}' not supported in forward method.")
        return images
