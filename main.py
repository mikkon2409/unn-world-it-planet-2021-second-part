import torch
import pdf2image
import cv2 as cv
import numpy as np
import pytesseract
import Levenshtein


detection_threshold = 0.8
dictionary = {'0': '0', '': '0', '9': '9', '5': '5', '1': '1', '3': '3', '7': '7', '8': '8',
              '6': '6', '17': '7', '2': '2', '04': '0', '4': '4', '03': '0', '63': '6', '13': '3',
              '10': '0', '73': '7', '43': '4', '33': '3', '91': '9', '44': '4', '23': '2',
              '18': '8', '19': '9', '53': '5', '12': '2', '24': '2', '74': '9', '14': '4',
              '01': '0', '84': '8', '21': '0', '35': '3', '68': '3', '15': '3', '41': '4',
              '83': '8', '07': '0', '51': '6', '16': '6', '40': '0', '103': '0', '94': '9'}

model_path = "maskrcnn_resnet50_water_meters.pth"

model = torch.load(model_path)
model.eval()


def get_bboxes(img_src):
    with torch.no_grad():
        image = cv.cvtColor(img_src, cv.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float).cpu()
        image = torch.unsqueeze(image, 0)

        outputs = model(image)
        
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
    return boxes


def pdf_or_png(file_path: str):
    with open(file_path, "rb") as file:
        format_bytes = file.read()[1:4]
        if format_bytes == b'PDF':
            return "PDF"
        elif format_bytes == b'PNG':
            return "PNG"


def get_image_from_filepath(filepath: str) -> np.ndarray:
    image = None
    file_type = pdf_or_png(filepath)
    if file_type == "PDF":
        images = pdf2image.convert_from_path(filepath, 300)
        image = cv.cvtColor(np.array(images[0]), cv.COLOR_RGB2BGR)
    elif file_type == "PNG":
        image = cv.imread(filepath, cv.IMREAD_COLOR)
    return image


def eliminate_escapes(src_str):
    escapes = ''.join([chr(char) for char in range(1, 32)])
    translator = str.maketrans('', '', escapes)
    return src_str.translate(translator)


def translate(src_str):
    if src_str in dictionary:
        return dictionary[src_str]
    output = list(map(lambda x: (x, Levenshtein.distance(src_str, x)), dictionary.keys()))
    output = min(output, key=lambda x: x[1])
    return dictionary[output[0]]


def set_point(src_str):
    chars = [char for char in src_str]
    chars.insert(5, '.')
    return ''.join(chars)


def recognize(img):
    config = "--psm 10 -c tessedit_char_whitelist=0123456789"
    img = cv.medianBlur(img, 5)
    img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
    h, w, _ = img.shape
    if w / h < 3:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        h, w, _ = img.shape

    step = w / 8
    output = []
    for i in range(7):
        img_to_recognize = img[:, int(step*i):int(step*(i+1)), :]
        symbol = pytesseract.image_to_string(img_to_recognize, config=config)
        output.append(symbol)
    output = list(map(lambda x: eliminate_escapes(x), output))
    output = list(map(lambda x: translate(x), output))
    if output[0] == '6':
        output[0] = '0'
    if output[1] == '6':
        output[0] = '0'

    output = ''.join(output)
    output = set_point(output)
    output = float(output)

    return output


def extract_image_features(filepath: str) -> dict:
    """
    Функция, которая будет вызвана для получения признаков документа, для которого
    задан:
    :param filepath: абсолютный путь до изображения на локальном компьютере (строго
    pdf или png).
    :return: возвращаемый словарь, совпадающий по составу и написанию ключей
    условию задачи
    """

    src_img = get_image_from_filepath(filepath)
    bboxes = get_bboxes(src_img)
    bbox = bboxes[0]
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    
    detected_box_img = src_img[y1:y2, x1:x2]
    prediction = recognize(detected_box_img)

    result = {
        'prediction': prediction,   # float, предсказанный показатель прибора с
                                    # точностью как минимум до 2х знаков после запятой

        'x1': x1,  # int, координата верхнего левого угла зоны показателей прибора
        'y1': y1,  # int, координата верхнего левого угла зоны показателей прибора

        'x2': x2,  # int, координата правого нижнего угла зоны показателей прибора
        'y2': y2,  # int, координата правого нижнего угла зоны показателей прибора
    }
    return result
