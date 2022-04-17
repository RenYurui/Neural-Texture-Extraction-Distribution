import torch

def crop_face_from_output(image, face_center, crop_smaller=0):
    r"""Crop out the face region of the image (and resize if necessary to feed
    into generator/discriminator).

    Args:
        image (NxC1xHxW tensor or list of tensors): Image to crop.
        input_label (list) list of the face center.
        crop_smaller (int): Number of pixels to crop slightly smaller region.
    Returns:
        output (NxC1xHxW tensor or list of tensors): Cropped image.
    """
    if type(image) == list:
        return [crop_face_from_output(im, face_center, crop_smaller)
                for im in image]
    output = None
    face_size = image.shape[-2] // 32 * 8
    for i in range(face_center.shape[0]):
        face_position = get_face_bbox_for_output(
            image,
            face_center[i],
            crop_smaller=crop_smaller)
        if face_position is not None:
            ys, ye, xs, xe = face_position
            output_i = torch.nn.functional.interpolate(
                image[i:i + 1, -3:, ys:ye, xs:xe],
                size=(face_size, face_size), mode='bilinear',
                align_corners=True)
        else:
            output_i = torch.zeros(1, 3, face_size, face_size, device=image.device)
        output = torch.cat([output, output_i]) if i != 0 else output_i
    return output


def get_face_bbox_for_output(image, face_center, crop_smaller=0):
    _,_,h,w = image.shape
    if torch.sum(face_center) != -4:
        xs, ys, xe, ye = face_center
        xc, yc = (xs + xe) // 2, (ys + ye) // 2
        ylen = int((xe - xs) * 2.5)

        ylen = xlen = min(w, max(32, ylen))
        yc = max(ylen // 2, min(h - 1 - ylen // 2, yc))
        xc = max(xlen // 2, min(w - 1 - xlen // 2, xc))

        ys, ye = int(yc) - ylen // 2, int(yc) + ylen // 2
        xs, xe = int(xc) - xlen // 2, int(xc) + xlen // 2
        if crop_smaller != 0:  # Crop slightly smaller region inside face.
            ys += crop_smaller
            xs += crop_smaller
            ye -= crop_smaller
            xe -= crop_smaller
        return [ys, ye, xs, xe]
    else:
        return None