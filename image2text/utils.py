import io
from PIL import Image

def byte_array_to_image(byte_data: bytes) -> Image:
    """
    Convert byte array to PIL Image.

    :param byte_data: Byte array representing an image.
    :return: PIL Image object.
    """
    return Image.open(io.BytesIO(byte_data))