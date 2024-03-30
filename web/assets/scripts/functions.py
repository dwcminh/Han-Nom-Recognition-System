import cv2
import numpy as np
from skimage.transform import resize
import math
import collections

NEG_INF = -float("inf")

def generate_initial_drawing(boxes, size_ratio):
    initial_drawing = {'version': '4.4.0', 'objects': []}
    for box_pts in boxes:
        left = max(box_pts[0][0], box_pts[3][0])
        top = max(box_pts[0][1], box_pts[1][1])
        width = max(box_pts[1][0], box_pts[2][0]) - min(box_pts[0][0], box_pts[3][0])
        height = max(box_pts[2][1], box_pts[3][1]) - min(box_pts[0][1], box_pts[1][1])
        
        initial_drawing['objects'].append({
            'type': 'rect',
            'left': left * size_ratio,
            'top': top * size_ratio,
            'width': width * size_ratio,
            'height': height * size_ratio,
            'fill': 'rgba(76, 175, 80, 0.3)',
            'stroke': 'red',
            'strokeWidth': 2,
            'strokeUniform': True,
            'transparentCorners': False
        })
    return initial_drawing


# https://github.com/andfanilo/streamlit-drawable-canvas/issues/65
def transform_fabric_box(box, size_ratio):
    scaled_width = box['width'] * box['scaleX']
    scaled_height = box['height'] * box['scaleY']
    return np.array([
        [box['left'], box['top']], 
        [box['left'] + scaled_width, box['top']], 
        [box['left'] + scaled_width, box['top'] + scaled_height], 
        [box['left'], box['top'] + scaled_height]
    ]) / size_ratio
    
    
def order_points_clockwise(box_points):
    points = np.array(box_points)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    quad_box = np.zeros((4, 2), dtype=np.float32)
    quad_box[0] = points[np.argmin(s)]
    quad_box[2] = points[np.argmax(s)]
    quad_box[1] = points[np.argmin(diff)]
    quad_box[3] = points[np.argmax(diff)]
    return quad_box


def order_boxes4nom(boxes):
    return sorted(boxes, key=lambda box: (
        box[:, 0].max(), 
        box[:, 1].min()
    ), reverse=True)
    

def get_patch(page, points):
    points = order_points_clockwise(points)
    page_crop_width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3]))
    )
    page_crop_height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2]))
    )
    pts_std = np.float32([
        [0, 0], [page_crop_width, 0], 
        [page_crop_width, page_crop_height],[0, page_crop_height]
    ])
    M = cv2.getPerspectiveTransform(points, pts_std)
    return cv2.warpPerspective(
        page, M, (page_crop_width, page_crop_height), 
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )
    
def distortion_free_resize(image, target_height=432, target_width=48, align_top=True):
    # Convert image to float32 numpy array if it's not already
    image = np.asarray(image, dtype=np.float32)

    # Calculate the aspect ratio of the target dimensions
    target_aspect = target_width / target_height
    # Calculate the aspect ratio of the input image
    image_height, image_width = image.shape[:2]
    image_aspect = image_width / image_height

    # Determine the scaling factor and the new dimensions of the image
    if image_aspect > target_aspect:
        # Image is wider than the target aspect ratio
        scale = target_width / image_width
        new_width = target_width
        new_height = np.round(image_height * scale).astype(int)
    else:
        # Image is taller than the target aspect ratio
        scale = target_height / image_height
        new_height = target_height
        new_width = np.round(image_width * scale).astype(int)

    # Resize the image using cv2
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate padding sizes
    pad_height = target_height - new_height
    pad_width = target_width - new_width

    # Apply padding to the resized image
    if align_top:
        pad_top = 0
        pad_bottom = pad_height
    else:
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Pad with white color (255 for each channel)
    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return padded_image

def resize_image_short_side(image, image_short_side=736):
    height, width, _ = image.shape
    if height < width:
        new_height = image_short_side
        new_width = int(round(new_height / height * width / 32) * 32)
    else:
        new_width = image_short_side
        new_height = int(round(new_width / width * height / 32) * 32)
    return cv2.resize(image, (new_width, new_height))