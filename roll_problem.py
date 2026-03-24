from PIL import Image, ExifTags

#
def fix_image_orientation(image_path):
    # openning image
    img = Image.open(image_path)

    # Check whether the image contains EXIF metadata
    try:
        # 获取EXIF信息
        exif = img._getexif()
        if exif is not None:
            # Traverse the EXIF information and search for the direction information
            for tag, value in exif.items():
                if tag == 274:  # 274 express direction
                    if value == 3:
                        img = img.rotate(180, expand=True)
                    elif value == 6:
                        img = img.rotate(270, expand=True)
                    elif value == 8:
                        img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # if don't have EXIF information，ignore
        pass

    return img


