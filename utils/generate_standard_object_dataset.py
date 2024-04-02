import cv2 as cv
import numpy as np 
import os
import random
import hashlib
from draw_standard_object import draw_shape 

BACKGROUND_SOURCE_IMG_PATH = os.path.abspath("assets/ground.png");
BACKGROUND_IMAGE = cv.imread(BACKGROUND_SOURCE_IMG_PATH)

BACKGROUND_ORIGINAL_HEIGHT, BACKGROUND_ORIGINAL_WIDTH = BACKGROUND_IMAGE.shape[:2]
RESULT_IMG_WIDTH = 1920
RESULT_IMG_HEIGHT = 1080 
SHAPE_IMG_SIZE = 150
#REL_BBOX_WIDTH = SHAPE_IMG_SIZE / BACKGROUND_ORIGINAL_WIDTH
#REL_BBOX_HEIGHT = SHAPE_IMG_SIZE / BACKGROUND_ORIGINAL_HEIGHT
REL_BBOX_SIZE = SHAPE_IMG_SIZE / BACKGROUND_ORIGINAL_HEIGHT

NUM_SAMPLES = 50000
PERCENT_TEST = .15
PERCENT_VALID = .15
PERCENT_TRAIN = .7

DATA_YAML_CONTENT = """names:
- CIRCLE
- SEMICIRCLE
- QUARTER_CIRCLE
- TRIANGLE
- RECTANGLE
- PENTAGON
- STAR
- CROSS
nc: 8
test: test/images 
train: train/images 
val: valid/images 
"""

OUTPUT_FOLDER = os.path.abspath("standard_object_dataset")
os.makedirs(OUTPUT_FOLDER)
os.makedirs(os.path.join(OUTPUT_FOLDER, "train"))
os.makedirs(os.path.join(OUTPUT_FOLDER, "train/images"))
os.makedirs(os.path.join(OUTPUT_FOLDER, "train/labels"))
os.makedirs(os.path.join(OUTPUT_FOLDER, "test"))
os.makedirs(os.path.join(OUTPUT_FOLDER, "test/images"))
os.makedirs(os.path.join(OUTPUT_FOLDER, "test/labels"))
os.makedirs(os.path.join(OUTPUT_FOLDER, "valid"))
os.makedirs(os.path.join(OUTPUT_FOLDER, "valid/images"))
os.makedirs(os.path.join(OUTPUT_FOLDER, "valid/labels"))


def get_random_background_segment():
    start_x = random.randint(0, BACKGROUND_ORIGINAL_WIDTH-RESULT_IMG_WIDTH) 
    start_y = random.randint(0, BACKGROUND_ORIGINAL_HEIGHT-RESULT_IMG_HEIGHT) 
    end_x = start_x + RESULT_IMG_WIDTH 
    end_y = start_y + RESULT_IMG_HEIGHT 

    segment = BACKGROUND_IMAGE[start_y:end_y, start_x:end_x]
    center = ((end_x - start_x) / 2, (end_y - start_y) / 2)
    rotation_matrix = cv.getRotationMatrix2D(center, random.randint(0, 360), 1.0)
    rotated_segment = cv.warpAffine(segment, rotation_matrix, (RESULT_IMG_WIDTH, RESULT_IMG_HEIGHT), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    return rotated_segment


def calculate_absolute_center(image, relative_center):
    height, width = image.shape[:2]
    absolute_x = int(width * relative_center[0])
    absolute_y = int(height * relative_center[1])
    return (absolute_x, absolute_y)


def create_transparent_image():
    return np.zeros((SHAPE_IMG_SIZE, SHAPE_IMG_SIZE, 4), dtype=np.uint8)


def rotate_shape_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv.warpAffine(image, rotation_matrix, (width, height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    new_image = np.zeros((height, width, 4), dtype=np.uint8)
    new_image[..., :3] = rotated_image[..., :3]
    new_image[..., 3] = cv.warpAffine(image[..., 3], rotation_matrix, (width, height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    
    return new_image



def overlay_shape_on_background(background, shape_img, relative_center):
    bg_height, bg_width = background.shape[:2]
    shape_height, shape_width = shape_img.shape[:2]

    center_x = int(relative_center[0] * bg_width)
    center_y = int(relative_center[1] * bg_height)
    top_left_x = center_x - shape_width // 2
    top_left_y = center_y - shape_height // 2
    y1, y2 = top_left_y, top_left_y + shape_height
    x1, x2 = top_left_x, top_left_x + shape_width
    alpha_s = shape_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    # blending 
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (alpha_s * shape_img[:, :, c] +
                                       alpha_l * background[y1:y2, x1:x2, c])

    return background


def get_safe_center(bg_width, bg_height, shape_width, shape_height):
    x_margin = shape_width / 2
    y_margin = shape_height / 2

    center_x = random.uniform(x_margin, bg_width - x_margin)
    center_y = random.uniform(y_margin, bg_height - y_margin)

    relative_center_x = center_x / bg_width
    relative_center_y = center_y / bg_height

    return (relative_center_x, relative_center_y)


def generate_filename(info_str, length):
    encoded_input = info_str.encode()
    hash_object = hashlib.sha256(encoded_input)
    hex_dig = hash_object.hexdigest()
    truncated_hash = hex_dig[:length]
    return truncated_hash


def generate_data(samples, dir):
    for _ in range(int(samples)):
        background_image = get_random_background_segment()

        standard_object = draw_shape(create_transparent_image()) 
        standard_object_image = rotate_shape_image(standard_object[0], random.randint(0, 360))
        standard_object_shape = standard_object[1][0]
        standard_object_shape_color = standard_object[1][2]
        standard_object_character = standard_object[1][1]
        standard_object_character_color = standard_object[1][3]
        standard_object_center = get_safe_center(RESULT_IMG_WIDTH, RESULT_IMG_HEIGHT, SHAPE_IMG_SIZE, SHAPE_IMG_SIZE) 

        standard_object_image_full = overlay_shape_on_background(background_image, standard_object_image, standard_object_center) 
        standard_object_label = f"{standard_object_shape} {standard_object_center[0]} {standard_object_center[1]} {REL_BBOX_SIZE} {REL_BBOX_SIZE}" 

        file_name = generate_filename(standard_object_label, 16) + "_" + standard_object_character + "_" + standard_object_shape_color + "_" + standard_object_character_color 
        print(file_name)
        img_path = os.path.join(OUTPUT_FOLDER, f"{dir}/images/{file_name}.jpg")
        label_path = os.path.join(OUTPUT_FOLDER, f"{dir}/labels/{file_name}.txt") 

        cv.imwrite(img_path, standard_object_image_full)
        with open(os.path.join(OUTPUT_FOLDER, label_path), "w") as f:
            f.write(standard_object_label)
        

def main():
    print("Generating train data...")
    generate_data(NUM_SAMPLES * PERCENT_TRAIN, "train")
    print("Generating test data...")
    generate_data(NUM_SAMPLES * PERCENT_TEST, "test")
    print("Generating validation data...")
    generate_data(NUM_SAMPLES * PERCENT_VALID, "valid")

    with open(os.path.join(OUTPUT_FOLDER, "data.yaml"), "w") as f:
        f.write(DATA_YAML_CONTENT)


if __name__ == "__main__":
    main()
