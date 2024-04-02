import cv2 as cv
import numpy as np
import random
import string
from PIL import ImageFont, ImageDraw, Image  


def draw_character(image, pos, character, character_color):
    size = 55 
    img = cv.cvtColor(image, cv.COLOR_BGRA2RGBA)  
    pil_img = Image.fromarray(img) 
    draw = ImageDraw.Draw(pil_img)  
    font = ImageFont.truetype("assets/font2.ttf", size)  
    draw.text((pos[0]-15, pos[1]-42), character, font=font, fill=(character_color[2], character_color[1], character_color[0], character_color[3]))  
    return cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGRA)  
    ###
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.7 
    thickness = 4 
    cv.putText(image, character, (pos[0]-20, pos[1]+15), font, font_scale, character_color, thickness, cv.LINE_AA)
    return image


def draw_circle(image, center, color, character, character_color):
    cv.circle(image, center, SHAPE_SIZE, color, thickness=-1, lineType=cv.LINE_AA)
    image = draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)+10), character, character_color)
    return image


def draw_semicircle(image, center, color, character, character_color):
    cv.ellipse(image, center, (SHAPE_SIZE+10, SHAPE_SIZE+10), 0, 0, 180, color, thickness=-1, lineType=cv.LINE_AA)
    image = draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)+35), character, character_color)
    return image


def draw_quarter_circle(image, center, color, character, character_color):
    cv.ellipse(image, (center[0]-(SHAPE_SIZE//2)-10, center[1]-(SHAPE_SIZE//2)-10), (SHAPE_SIZE*2, SHAPE_SIZE*2), 0, 0, 90, color, thickness=-1, lineType=cv.LINE_AA)
    image = draw_character(image, (SHAPE_IMG_SIZE//2+5, (SHAPE_IMG_SIZE//2)+15), character, character_color)
    return image


def draw_triangle(image, center, color, character, character_color):
    vertices = np.array([
        [center[0] - SHAPE_SIZE, center[1] + SHAPE_SIZE],
        [center[0], center[1] - SHAPE_SIZE],
        [center[0] + SHAPE_SIZE, center[1] + SHAPE_SIZE]
    ], np.int32)
    cv.fillPoly(image, [vertices], color, lineType=cv.LINE_AA)
    image = draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)+15), character, character_color)
    return image


def draw_rectangle(image, center, color, character, character_color):
    top_left = (center[0] - SHAPE_SIZE, center[1] - int(SHAPE_SIZE / 2))
    bottom_right = (center[0] + SHAPE_SIZE, center[1] + int(SHAPE_SIZE / 2) + 10)
    cv.rectangle(image, top_left, bottom_right, color, -1, lineType=cv.LINE_AA)
    image = draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)+10), character, character_color)
    return image


def draw_pentagon(image, center, color, character, character_color):
    pts = np.array([[np.cos(theta) * SHAPE_SIZE + center[0], np.sin(theta) * SHAPE_SIZE + center[1]] 
                    for theta in np.linspace(0, 2 * np.pi, 6)[:-1] + np.pi / 5])
    cv.fillPoly(image, [np.array(pts, np.int32)], color, lineType=cv.LINE_AA)
    image = draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)+5), character, character_color)
    return image


def draw_star(image, center, color, character, character_color):
    num_vertices = 5

    outer_radius = SHAPE_SIZE + 18 
    inner_radius = outer_radius * np.cos(2 * np.pi / num_vertices) / np.cos(np.pi / num_vertices)
    outer_angle = 2 * np.pi / num_vertices
    inner_angle = np.pi / num_vertices

    points = []

    for i in range(num_vertices):
        outer_x = int(center[0] + outer_radius * np.cos(outer_angle * i))
        outer_y = int(center[1] - outer_radius * np.sin(outer_angle * i))
        inner_x = int(center[0] + inner_radius * np.cos(outer_angle * i + inner_angle))
        inner_y = int(center[1] - inner_radius * np.sin(outer_angle * i + inner_angle))
        points.append((outer_x, outer_y))
        points.append((inner_x, inner_y))

    star_points = np.array(points, np.int32).reshape((-1, 1, 2))
    cv.fillPoly(image, [star_points], color, lineType=cv.LINE_AA)

    image = draw_character(image, (SHAPE_IMG_SIZE//2-2, (SHAPE_IMG_SIZE//2)+5), character, character_color)

    return image


def draw_cross(image, center, color, character, character_color):
    thickness = (SHAPE_SIZE // 2) + 16 

    start_point_h = (center[0] - SHAPE_SIZE, center[1] - thickness // 2)
    end_point_h = (center[0] + SHAPE_SIZE, center[1] + thickness // 2)
    cv.rectangle(image, start_point_h, end_point_h, color, -1)

    start_point_v = (center[0] - thickness // 2, center[1] - SHAPE_SIZE)
    end_point_v = (center[0] + thickness // 2, center[1] + SHAPE_SIZE)
    cv.rectangle(image, start_point_v, end_point_v, color, -1)

    image = draw_character(image, (SHAPE_IMG_SIZE//2, (SHAPE_IMG_SIZE//2)+3), character, character_color)

    return image


SHAPE_SIZE = 50
SHAPE_IMG_SIZE = 150

DRAW_SHAPE_FUNCTIONS = {
    0: draw_circle,
    1: draw_semicircle,
    2: draw_quarter_circle,
    3: draw_triangle,
    4: draw_rectangle,
    5: draw_pentagon,
    6: draw_star,
    7: draw_cross
}

COLORS = {
    "white": (255, 255, 255, 255),
    "black": (0, 0, 0, 255),
    "red": (0, 0, 255, 255),
    "blue": (255, 0, 0, 255),
    "green": (0, 255, 0, 255),
    "purple": (128, 0, 128, 255),
    "brown": (19, 69, 139, 255),
    "orange": (0, 165, 255, 255)
}

def draw_shape(image):
    center = (SHAPE_IMG_SIZE//2, SHAPE_IMG_SIZE//2)

    character = random.choice(string.ascii_uppercase + string.digits)
    shape = random.randint(0, 7)

    shape_color = random.choice(list(COLORS)) 
    character_color = random.choice(list(COLORS))
    while character_color == shape_color:
        character_color = random.choice(list(COLORS))

    draw_function = DRAW_SHAPE_FUNCTIONS.get(shape, None)
    standard_object_image = draw_function(image, center, COLORS[shape_color], character, COLORS[character_color])
    standard_object_info = (shape, character, shape_color, character_color)

    return (standard_object_image, standard_object_info)
    