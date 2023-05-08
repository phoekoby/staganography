import numpy as np
import skimage.io
from skimage import io
from skimage.util import view_as_blocks
from scipy.fftpack import dct, idct

blue_index=2
# Координаты для изменения коэффициентов DCT.
u1, v1 = 4, 5
u2, v2 = 5, 4
n = 8  # размер блока.
P = 25  # Пороговое значение для сравнения разницы коэффициентов.


# Увеличение коэффициента.
def increment_abs(x):
    return x + 1 if x >= 0 else x - 1


# Уменьшение коэффициента.
def decrement_abs(x):
    if np.abs(x) <= 1:
        return 0
    else:
        return x - 1 if x >= 0 else x + 1


# Разница между коэффициентами.
def abs_diff_coefs(transform):
    return abs(transform[u1, v1]) - abs(transform[u2, v2])


# Проверка условия.
def valid_coefficients(transform, bit, threshold):
    difference = abs_diff_coefs(transform)
    if (bit == 0) and (difference > threshold):
        return True
    elif (bit == 1) and (difference < -threshold):
        return True
    else:
        return False


# Изменение коэффициентов.
def change_coefficients(transform, bit):
    coefs = transform.copy()
    if bit == 0:
        coefs[u1, v1] = increment_abs(coefs[u1, v1])
        coefs[u2, v2] = decrement_abs(coefs[u2, v2])
    elif bit == 1:
        coefs[u1, v1] = decrement_abs(coefs[u1, v1])
        coefs[u2, v2] = increment_abs(coefs[u2, v2])
    return coefs


def double_to_byte(arr):
    return np.uint8(np.round(np.clip(arr, 0, 255), 0))


# Вставка бита в блок пискселей.
def hide_bit(block, bit):
    patch = block.copy()
    coefs = dct(dct(patch, axis=0), axis=1)
    while not valid_coefficients(coefs, bit, P) or (bit != get_message_from_bit(patch)):
        coefs = change_coefficients(coefs, bit)
        patch = double_to_byte(idct(idct(coefs, axis=0), axis=1) / (2 * n) ** 2)
    return patch


# Вставка сообщения в изображение.
def hide_message(orig, msg):
    changed = orig.copy()
    blue = changed[:, :, 2]
    blocks = view_as_blocks(blue, block_shape=(n, n))
    h = blocks.shape[1]
    for index, bit in enumerate(msg):
        i = index // h
        j = index % h
        block = blocks[i, j]
        blue[i * n: (i + 1) * n, j * n: (j + 1) * n] = hide_bit(block, bit)
    changed[:, :, 2] = blue
    return changed


# Получение бита.
def get_message_from_bit(block):
    transform = dct(dct(block, axis=0), axis=1)
    return 0 if abs_diff_coefs(transform) > 0 else 1


# Получение сообщения.
def get_message_from_image(img, length):
    blocks = view_as_blocks(img[:, :, 2], block_shape=(n, n))
    h = blocks.shape[1]
    return [get_message_from_bit(blocks[index // h, index % h]) for index in range(length)]


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


# Метод для подсчета операции Лаплласа
def lapllas(matrix):
    res = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            s = 0
            amount = 0
            if i >= 0:
                s += matrix[i - 1][j]
                amount += 1
            if j >= 0:
                s += matrix[i][j - 1]
                amount += 1
            if i < matrix.shape[0] - 1:
                s += matrix[i + 1][j]
                amount += 1
            if j < matrix.shape[1] - 1:
                s += matrix[i][j + 1]
                amount += 1
            s -= amount * matrix[i][j]
            res[i][j] = s
    return res


# Метод для подсчета метрик
def metrics(empty, full):
    empty = empty[:, :, blue_index]
    full = full[:, :, blue_index]
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
    print(f"Пиковое отношение сигнал-шум: %f" % m_PSNR)
    # Подсчет Среднего квадратичного отклонения по формуле
    m_MSE = (1 / (W * H)) * np.sum((empty - full) * (empty - full))
    print(f"Среднее квадратичное отклонение: %f" % m_MSE)

    # Подсчет Среднего квадратичного отклонения лапласиана
    empty_lapllas = lapllas(empty)
    full_laplas = lapllas(full)
    m_LMSE = np.sum((empty_lapllas - full_laplas) ** 2) / np.sum(empty_lapllas * empty_lapllas)
    print(f"Среднее квадратичное отклонение лапласиана: %f" % m_LMSE)


if __name__ == "__main__":
    # Чтение картинки
    image = read_image("sea.jpg")
    # Чтение сообщения из файла message.txt
    lines = []
    with open("message.txt", 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line)
    text = ''.join(lines)
    # Преобразование сообщения в массив битов
    bites = to_bits(text)

    energy = 1

    omega = 5
    key = 100
    # Получение заполненного стегокнтейнера
    encoded_image = hide_message(image, bites)
    skimage.io.imsave("encoded-sea.jpg", encoded_image)

    # Получение соолбщения из заполненного стегокнтейнера
    decoded_bits_message = get_message_from_image(encoded_image, len(bites))

    # Вывод строки из полученного массива битов сообщения
    print(from_bits(decoded_bits_message))

    with open("message-from-image.txt", 'w', encoding='utf8') as f:
        f.write(from_bits(decoded_bits_message))
    # подсчет метрик на основе незаполненного и заполненного контейнеров
    metrics(image, encoded_image)
