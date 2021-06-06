import Levenshtein
import torch
import pdf2image
import cv2 as cv
import numpy as np
import pytesseract
from math import floor
from Levenshtein import distance as lv_distance


detection_threshold = 0.8
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


def truncate(f, n):
    return floor(f * 10 ** n) / 10 ** n


def eliminate_escapes(src_str):
    escapes = ''.join([chr(char) for char in range(1, 32)])
    translator = str.maketrans('', '', escapes)
    return src_str.translate(translator)


def translate(src_str):
    from_char = 'liIgA|)soODp'
    to_char = '111981150000'
    table = str.maketrans(from_char, to_char)
    return src_str.translate(table)


digits = '0123456789'


def clean_str(src_str):
    return ''.join(list(filter(lambda x: x in digits, src_str)))


def set_point(src_str):
    chars = [char for char in src_str]
    chars.insert(5, '.')
    return ''.join(chars)


def eliminate_trash(src_str):
    if len(src_str) == 1:
        if src_str in digits:
            return src_str
        else:
            res = translate(src_str)

    elif len(src_str) == 2:
        pass
    elif len(src_str) == 3:
        pass
    else:
        pass
    return ''


comparison_dict = {}


def make_translator():
    res_dict = {}
    for i, k in comparison_dict.items():
        res_dict[i] = str(np.argmax(k))
    return res_dict


def compare2dict(my, gt):
    for i, j in zip(my, gt):
        if i not in comparison_dict:
            comparison_dict[i] = [0 for _ in range(10)]
        comparison_dict[i][int(j)] += 1


def recognize(img, gt_value):
    config = "--psm 10 -c tessedit_char_whitelist=0123456789"
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img, 5)
    img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
    _, w, _ = img.shape
    step = w // 8
    output = []
    for i in range(7):
        img_to_recognize = img[:, step * i:step * (i + 1), :]
        symbol = pytesseract.image_to_string(img_to_recognize, config=config)
        output.append(symbol)
    output = list(map(lambda x: eliminate_escapes(x), output))
    compare2dict(output, gt_value)

    # output = list(map(lambda x: '0' if x == '' else x, output))
    # output = list(map(lambda x: eliminate_trash(x), output))
    #
    # cv.imshow('img', img)
    # cv.waitKey(0)
    #
    # output = clean_str(output)
    #
    # output = set_point(output)
    # output = float(output)
    output = 1.0
    # output = truncate(output, 2)
    # print(output)
    return output


def gt_value_to_list(gt_value):
    value = str(gt_value)
    l, r = value.split(sep=".")
    if len(l) < 5:
        l = l.zfill(5)
    if len(r) < 3:
        r = r + '0' * (3 - len(r))

    return [*l, *r][:7]


def extract_image_features(filepath, gt_value):
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
    gt_value_list = gt_value_to_list(gt_value)
    prediction = recognize(detected_box_img, gt_value_list)

    result = {
        'prediction': prediction,  # float, предсказанный показатель прибора с
        # точностью как минимум до 2х знаков после запятой

        'x1': x1,  # int, координата верхнего левого угла зоны показателей прибора
        'y1': y1,  # int, координата верхнего левого угла зоны показателей прибора

        'x2': x2,  # int, координата правого нижнего угла зоны показателей прибора
        'y2': y2,  # int, координата правого нижнего угла зоны показателей прибора
    }
    return result


if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import csv

    print(Levenshtein.distance("aaa", "aaa"))
    print(Levenshtein.distance("aba", "aaa"))
    print(Levenshtein.distance("aba", "baa"))
    print(Levenshtein.distance("ab", "ca"))

    # path_to_png = "TlkWaterMeters/png"
    # with open("TlkWaterMeters/data.tsv") as data:
    #     dataset = csv.reader(data, delimiter="\t")
    #     dataset = list(dataset)[1:]
    #     # dataset = dataset[:20]
    #     count = 0
    #     for row in tqdm(dataset):
    #         path_to_img = os.path.join(path_to_png, row[0].rstrip('.jpg') + '.png')
    #         gt_value = row[1]
    #         # print(gt_value)
    #         # print(gt_value_to_list(gt_value))
    #         results = extract_image_features(path_to_img, gt_value)
    #         # my_value = results['prediction']
    #         # if my_value == gt_value:
    #         #     count += 1
    #     print(comparison_dict)
    #     print(make_translator())
    #     print("Recognized correctly:", count, "Total:", len(dataset))
