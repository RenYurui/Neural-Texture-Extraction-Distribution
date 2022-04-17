import os

def save_pilimage_in_jpeg(fullname, output_img):
    r"""Save PIL Image to JPEG.

    Args:
        fullname (str): Full save path.
        output_img (PIL Image): Image to be saved.
    """
    dirname = os.path.dirname(fullname)
    os.makedirs(dirname, exist_ok=True)
    output_img.save(fullname, 'JPEG', quality=99)
