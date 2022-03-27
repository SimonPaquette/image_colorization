import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np
from keras.models import load_model

WIDTH = 128
HEIGHT = 128
SIZE = (WIDTH, HEIGHT)
BATCH = 8


def manage_arg():
    parser = argparse.ArgumentParser(description="COLOR")
    parser.add_argument("path", type=Path, help="dir of file")
    parser.add_argument("model_path", type=Path, help="h5 model")
    args = parser.parse_args()
    return args


def pathlist_from_dir(dir: Path):
    pathnames = list(dir.iterdir())
    return pathnames


def preprocess_image(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Open, resize, convert to LAB, and normalize the image

    Args:
        image_path (Path): image to be opened

    Returns:
        Tuple[np.ndarray, np.ndarray]: L channel and AB channel
    """
    image = cv.imread(str(image_path), cv.IMREAD_COLOR)
    image = cv.resize(image, SIZE)
    image = image.astype(np.uint8)
    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    image = image.astype(np.float32)
    image /= 255.0
    l, a, b = cv.split(image)
    ab = cv.merge([a, b])
    return (l, ab)


def predict_ab_channel(l_channel: np.ndarray, model) -> np.ndarray:
    """
    Color prediction (ab_channel) from an image (using the grayscale L*) created by the model

    Args:
        l_channel (np.ndarray): input
        model (_type_): a TF model for colorization

    Returns:
        np.ndarray: a new predicted ab channel
    """
    grayscale = l_channel.reshape(1, HEIGHT, WIDTH, 1)
    ab_prediction = model.predict(grayscale, batch_size=BATCH)
    ab_prediction = ab_prediction.reshape(HEIGHT, WIDTH, 2)
    return ab_prediction


def construct_image(l_channel: np.ndarray, ab_channel: np.ndarray) -> np.ndarray:
    """
    Merge LAB, clip, range value between 0-255, and convert to RGB

    Args:
        l_channel (np.ndarray): L* lightness
        ab_channel (np.ndarray): a* green-red && b* yellow-blue

    Returns:
        np.ndarray: RGB image
    """
    image = cv.merge([l_channel, ab_channel])
    image = np.clip(image, 0, 1)
    image *= 255
    image = image.astype(np.uint8)
    image = cv.cvtColor(image, cv.COLOR_LAB2BGR)  # FOR OPENCV
    return image


def main():
    args = manage_arg()

    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    model = load_model(args.model_path)
    model.summary()

    path = args.path
    images_path = []
    output_directory = Path("")
    if not path.exists():
        raise FileNotFoundError(path)
    elif path.is_file():
        images_path.append(path)
        output_directory = path.parent
    elif path.is_dir():
        images_path = pathlist_from_dir(path)
        output_directory = Path(path.parent, f"{path.stem}_color{path.suffix}")
        output_directory.mkdir(exist_ok=True)

    print("Output directory:", output_directory)
    for i, image_path in enumerate(images_path):
        l_channel, _ = preprocess_image(image_path)
        ab_channel = predict_ab_channel(l_channel, model)
        image = construct_image(l_channel, ab_channel)

        path = Path(output_directory, f"{image_path.stem}_color{image_path.suffix}")

        if i == 0 or i == len(images_path) - 1:
            print(f"image {i+1} of {len(images_path)}, {path}")
        else:
            print(f"image {i+1} of {len(images_path)}, {path.name}        ", end="\r")

        cv.imwrite(str(path), image)


def visualizeLAB():
    length = 256

    L0 = np.full((length, length), 0).astype(np.uint8)
    L64 = np.full((length, length), 64).astype(np.uint8)
    L128 = np.full((length, length), 128).astype(np.uint8)
    L192 = np.full((length, length), 192).astype(np.uint8)
    L255 = np.full((length, length), 255).astype(np.uint8)

    Ax = [[x for x in range(length)]] * 256
    Ax = np.array(Ax).astype(np.uint8)

    Ay = []
    for x in range(length):
        Ay.append([x] * 256)
    Ay = np.array(Ay).astype(np.uint8)

    imageL0 = cv.merge([L0, Ax, Ay])
    imageL0 = cv.cvtColor(imageL0, cv.COLOR_LAB2BGR)
    cv.imwrite("L0_Ax_By.png", imageL0)

    imageL64 = cv.merge([L64, Ax, Ay])
    imageL64 = cv.cvtColor(imageL64, cv.COLOR_LAB2BGR)
    cv.imwrite("L64_Ax_By.png", imageL64)

    imageL128 = cv.merge([L128, Ax, Ay])
    imageL128 = cv.cvtColor(imageL128, cv.COLOR_LAB2BGR)
    cv.imwrite("L128_Ax_By.png", imageL128)

    imageL192 = cv.merge([L192, Ax, Ay])
    imageL192 = cv.cvtColor(imageL192, cv.COLOR_LAB2BGR)
    cv.imwrite("L192_Ax_By.png", imageL192)

    imageL255 = cv.merge([L255, Ax, Ay])
    imageL255 = cv.cvtColor(imageL255, cv.COLOR_LAB2BGR)
    cv.imwrite("L255_Ax_By.png", imageL255)


if __name__ == "__main__":
    main()
    # visualizeLAB()
