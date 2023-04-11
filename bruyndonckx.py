import math
import random

import numpy as np
import skimage.io

blue_index = 2
green_index = 1
red_index = 0

block_size = 8
alpha = 0.02


def green(RGB):
    return RGB[green_index]


def blue(RGB):
    return RGB[blue_index]


def red(RGB):
    return RGB[red_index]


def L(RGB):
    return int(0.299 * red(RGB) + 0.587 * green(RGB) + 0.114 * blue(RGB))

def extract_blue_channel(picture):
    blue_channel = picture[:, :, 2]
    blue_img = np.zeros(picture.shape)
    blue_img[:, :, 2] = blue_channel
    return blue_img


def br_mask(image):
    a = np.zeros((block_size ** 2, 3))
    n = 0
    for i in range(block_size):
        for j in range(block_size):
            a[n][0] = i
            a[n][1] = j
            r, g, b = image[j, i, 0:3]
            a[n][2] = r * 0.299 + g * 0.587 + b * 0.114
            n += 1

    a = a[np.argsort(a[:, 2])]
    b_mask = np.zeros((block_size, block_size))
    for b in a[0:math.ceil(block_size ** 2 / 2)]:
        i = math.floor(b[0])
        j = math.floor(b[1])
        b_mask[i][j] = 1
    return b_mask


def mask():
    m = np.zeros((block_size, block_size))
    for i in range(block_size):
        for j in range(block_size):
            m[i][j] = random.randrange(0, 2, 1)
    return m


def brightness(image, mask):
    br = 0
    c = np.sum(mask)
    for i in range(block_size):
        for j in range(block_size):
            r, g, b = image[j, i, 0:3]
            pix_br = r * 0.299 + g * 0.587 + b * 0.114
            if mask[i][j] == 1:
                br += pix_br
    br = br / c
    return br


def add_brightness(image, mask):
    for i in range(block_size):
        for j in range(block_size):
            if mask[i, j] <= 0:
                continue
            r, g, b = image[j, i, 0:3] + 1 / 255
            if r > 1:
                r = 1
            if g > 1:
                g = 1
            if b > 1:
                b = 1
            image[j, i, 0:3] = r, g, b
    return image


def remove_brightness(image, mask):
    for i in range(block_size):
        for j in range(block_size):
            if mask[i, j] <= 0:
                continue
            r, g, b = image[j, i, 0:3] - 1 / 255
            if r < 0:
                r = 0
            if g < 0:
                g = 0
            if b < 0:
                b = 0
            image[j, i, 0:3] = r, g, b
    return image


def hide_message(message_bits, img, l_b, key):
    img = img.copy()
    random.seed(key)
    r_mask = mask()
    m_i = 0

    w = img.shape[1]
    h = img.shape[0]

    w_b = math.floor(w / block_size)
    h_b = math.floor(h / block_size)

    for i in range(w_b):
        for j in range(h_b):
            if m_i >= l_b:
                break

            x = i * block_size
            y = j * block_size

            block_mask = br_mask(img[y:y + block_size, x:x + block_size, :])

            mask_1_a = np.logical_and(r_mask, np.logical_not(block_mask))
            mask_2_a = np.logical_and(np.logical_not(r_mask), np.logical_not(block_mask))
            mask_1_b = np.logical_and(r_mask, block_mask)
            mask_2_b = np.logical_and(np.logical_not(r_mask), block_mask)

            l_1_a = brightness(img[y:y + block_size, x:x + block_size, :], mask_1_a)
            l_2_a = brightness(img[y:y + block_size, x:x + block_size, :], mask_2_a)
            l_1_b = brightness(img[y:y + block_size, x:x + block_size, :], mask_1_b)
            l_2_b = brightness(img[y:y + block_size, x:x + block_size, :], mask_2_b)

            bit = message_bits[m_i]
            if bit == 1:
                while l_1_a <= l_2_a or abs(l_1_a - l_2_a) < alpha:
                    img[y:y + block_size, x:x + block_size, :] = add_brightness(
                        img[y:y + block_size, x:x + block_size, :], mask_1_a)
                    img[y:y + block_size, x:x + block_size, :] = remove_brightness(
                        img[y:y + block_size, x:x + block_size, :], mask_2_a)
                    l_1_a = brightness(img[y:y + block_size, x:x + block_size, :], mask_1_a)
                    l_2_a = brightness(img[y:y + block_size, x:x + block_size, :], mask_2_a)

                while l_1_b <= l_2_b or abs(l_2_a - l_2_b) < alpha:
                    img[y:y + block_size, x:x + block_size, :] = add_brightness(
                        img[y:y + block_size, x:x + block_size, :], mask_1_b)
                    img[y:y + block_size, x:x + block_size, :] = remove_brightness(
                        img[y:y + block_size, x:x + block_size, :], mask_2_b)
                    l_1_b = brightness(img[y:y + block_size, x:x + block_size, :], mask_1_b)
                    l_2_b = brightness(img[y:y + block_size, x:x + block_size, :], mask_2_b)

            else:
                while l_1_a >= l_2_a or abs(l_1_a - l_2_a) < alpha:
                    img[y:y + block_size, x:x + block_size, :] = remove_brightness(
                        img[y:y + block_size, x:x + block_size, :], mask_1_a)
                    img[y:y + block_size, x:x + block_size, :] = add_brightness(
                        img[y:y + block_size, x:x + block_size, :], mask_2_a)
                    l_1_a = brightness(img[y:y + block_size, x:x + block_size, :], mask_1_a)
                    l_2_a = brightness(img[y:y + block_size, x:x + block_size, :], mask_2_a)

                while l_1_b >= l_2_b or abs(l_1_b - l_2_b) < alpha:
                    img[y:y + block_size, x:x + block_size, :] = remove_brightness(
                        img[y:y + block_size, x:x + block_size, :], mask_1_b)
                    img[y:y + block_size, x:x + block_size, :] = add_brightness(
                        img[y:y + block_size, x:x + block_size, :], mask_2_b)
                    l_1_b = brightness(img[y:y + block_size, x:x + block_size, :], mask_1_b)
                    l_2_b = brightness(img[y:y + block_size, x:x + block_size, :], mask_2_b)

            m_i += 1

        if m_i >= l_b:
            break
    return img


# Метод получения сообщения из картинки
def get_message_from_image(picture, l_b, key):
    random.seed(key)
    r_mask = mask()

    w = picture.shape[1]
    h = picture.shape[0]

    w_b = math.floor(w / block_size)
    h_b = math.floor(h / block_size)
    m_i = 0
    res_bits = []
    for i in range(w_b):
        for j in range(h_b):
            if m_i >= l_b:
                break

            x = i * block_size
            y = j * block_size

            block_mask = br_mask(picture[y:y + block_size, x:x + block_size, :])

            mask_1_a = np.logical_and(r_mask, np.logical_not(block_mask))
            mask_2_a = np.logical_and(np.logical_not(r_mask), np.logical_not(block_mask))
            mask_1_b = np.logical_and(r_mask, block_mask)
            mask_2_b = np.logical_and(np.logical_not(r_mask), block_mask)

            l_1_a = brightness(picture[y:y + block_size, x:x + block_size, :], mask_1_a)
            l_2_a = brightness(picture[y:y + block_size, x:x + block_size, :], mask_2_a)
            l_1_b = brightness(picture[y:y + block_size, x:x + block_size, :], mask_1_b)
            l_2_b = brightness(picture[y:y + block_size, x:x + block_size, :], mask_2_b)

            bit = 0

            da = l_1_a - l_2_a
            db = l_1_b - l_2_b

            if (abs(da) < abs(db) and db > 0) or (abs(da) > abs(db) and db > 0):
                bit = 1
            res_bits.append(bit)
            m_i += 1

        if m_i >= l_b:
            break
    return res_bits


# Чтение картинки из файла
def read_image(path):
    image = skimage.io.imread(path)
    return image


# Преобразование строки в массив бит
def to_bits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


# Преобразование массива бит в строку
def from_bits(bits):
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


# Метод для подсчета метрик
def metrics(empty, full):
    empty = empty
    full = full
    # Подсчет Максимального абсолютного отклонения по формуле
    max_d = np.max(np.abs(empty.astype(int) - full.astype(int)))
    print(f"Максимальное абсолютное отклонение: %d" % max_d)
    # Подсчет Нормированного среднее квадратичного отклонения по формуле
    m_NMSE = np.sum((empty - full) * (empty - full)) / np.sum((empty * empty))
    print(f"Нормированное среднее квадратичное отклонение: %f" % m_NMSE)
    # Подсчет Отношения сигнал-шум по формуле
    m_SNR = 1 / m_NMSE
    print(f"Отношение сигнал-шум: %f" % m_SNR)
    # Подсчет Пикового отношения сигнал-шум по формуле
    H = empty.shape[0]
    W = empty.shape[1]
    m_PSNR = W * H * ((np.max(empty) ** 2) / np.sum((empty - full) * (empty - full)))
    print(f"Пиковое отношение сигнал-шум : %f" % m_PSNR)


if __name__ == "__main__":
    # Чтение картинки
    image = read_image("cat.jpg")
    # Чтение сообщения из файла message.txt
    lines = []
    with open("message.txt", 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line)
    text = ''.join(lines)
    # Преобразование сообщения в массив битов
    bites = to_bits(text)

    skimage.io.imsave("before-blue-channel.jpg", extract_blue_channel(image))
    key = 1000
    # Получение заполненного стегокнтейнера
    encoded_image = hide_message(bites, image, len(bites), key=key)
    skimage.io.imsave("after-blue-channel.jpg", extract_blue_channel(encoded_image))
    skimage.io.imsave("encoded-cat.jpg", encoded_image)
    # Получение соолбщения из заполненного стегокнтейнера

    decoded_bits_message = get_message_from_image(encoded_image, len(bites), key=key)
    # Вывод строки из полученного массива битов сообщения
    print(from_bits(decoded_bits_message))

    with open("message-from-image.txt", 'w', encoding='utf8') as f:
        f.write(from_bits(decoded_bits_message))
    # подсчет метрик на основе незаполненного и заполненного контейнеров
    metrics(image, encoded_image)
