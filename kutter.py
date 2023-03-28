import numpy as np
import skimage.io

blue_index = 2
green_index = 1
red_index = 0


def green(RGB):
    return RGB[green_index]


def blue(RGB):
    return RGB[blue_index]


def red(RGB):
    return RGB[red_index]


def L(RGB):
    return int(0.299 * red(RGB) + 0.587 * green(RGB) + 0.114 * blue(RGB))


def I(omega, picture, x, y):
    height = picture.shape[0]
    width = picture.shape[1]
    res = 0
    for j in range(1, omega + 1):
        if y + j < width:
            res += blue(picture[x, y + j])
        if y - j >= 0:
            res += blue(picture[x, y - j])
        if x + j < height:
            res += blue(picture[x + j, y])
        if x - j >= 0:
            res += blue(picture[x - j, y])
    return (1. / (4. * omega)) * res


def extract_blue_channel(picture):
    blue_channel = picture[:, :, 2]
    blue_img = np.zeros(picture.shape)
    blue_img[:, :, 2] = blue_channel
    return blue_img


# Метод сокрытия сообщения в картинке
def hide_message(message_bits, picture, l, omega):
    print(message_bits[:20])
    # Проверка вместимости стегоконтейнера, по сравнению с длинной сообщения
    if len(message_bits) > (picture.shape[0] - omega) * (picture.shape[1] - omega):
        print("I can not hide message")
        return picture
    i, j = omega, omega
    image = np.copy(picture)
    for bit in message_bits:
        if j == image.shape[1] - omega:
            j = omega
            i += 1
        if bit == 0:
            blue_ = int(min(255, blue(image[i][j]) + l * L(image[i][j])))
        else:
            blue_ = int(max(0, blue(image[i][j]) - l * L(image[i][j])))
        image[i][j][blue_index] = blue_
        j += 1
    return image


# Метод получения сообщения из картинки
def get_message_from_image(picture, omega, N):
    res_bites = []
    i, j = omega, omega
    for m in range(N):
        if j == picture.shape[1] - omega:
            j = omega
            i += 1
        if picture[i][j][blue_index] <= I(omega, picture, i, j):
            res_bites.append(1)
        else:
            res_bites.append(0)
        j += 1
    print(res_bites[:20])
    return res_bites


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

    energy = 0.05
    skimage.io.imsave("before-blue-channel.jpg", extract_blue_channel(image))

    omega = 3
    # Получение заполненного стегокнтейнера
    encoded_image = hide_message(bites, image, energy, omega=omega)
    skimage.io.imsave("after-blue-channel.jpg", extract_blue_channel(encoded_image))
    skimage.io.imsave("encoded-cat.jpg", encoded_image)
    # Получение соолбщения из заполненного стегокнтейнера

    decoded_bits_message = get_message_from_image(encoded_image, omega, len(bites))
    # Вывод строки из полученного массива битов сообщения
    print(from_bits(decoded_bits_message))

    with open("message-from-image.txt", 'w', encoding='utf8') as f:
        f.write(from_bits(decoded_bits_message))
    # подсчет метрик на основе незаполненного и заполненного контейнеров
    metrics(image, encoded_image)