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

# Значение яркости
def L(RGB):
    return int(0.299 * red(RGB) + 0.587 * green(RGB) + 0.114 * blue(RGB))


# Нахождение I*, идем во все четыре стороны и прибавляем значение голубого байта
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

#Получение картинки в синем канале
def extract_blue_channel(picture):
    blue_channel = picture[:, :, 2]
    blue_img = np.zeros(picture.shape)
    blue_img[:, :, 2] = blue_channel
    return blue_img

#Генерация уникальных координат пикселя для встраивания
def generate_unique_x_y(before, height, width):
    x = int(np.random.rand() * (height - 2 * omega) + omega)
    y = int(np.random.rand() * (width - 2 * omega) + omega)
    while (x, y) in before:
        x = int(np.random.rand() * (height - 2 * omega) + omega)
        y = int(np.random.rand() * (width - 2 * omega) + omega)
    return x, y


# Метод сокрытия сообщения в картинке
def hide_message(message_bits, picture, l, omega, key):
    # Проверка вместимости стегоконтейнера, по сравнению с длинной сообщения
    if len(message_bits) > (picture.shape[0] - omega) * (picture.shape[1] - omega):
        print("I can not hide message")
        return picture

    height = picture.shape[0]
    width = picture.shape[1]
    # Инициализируем рандом ключом
    np.random.seed(key)

    #Использованные координаты
    before = []
    image = np.copy(picture)
    # Идем по каждому биту
    for bit in message_bits:
        # Генерируем координаты
        x, y = generate_unique_x_y(before, height, width)
        before.append((x, y))
        #Встраиваем
        if bit == 0:
            blue_ = int(min(255, blue(image[x][y]) + l * L(image[x][y])))
        else:
            blue_ = int(max(0, blue(image[x][y]) - l * L(image[x][y])))
        image[x][y][blue_index] = blue_

    return image


# Метод получения сообщения из картинки
def get_message_from_image(picture, omega, N, key):
    res_bites = []

    height = picture.shape[0]
    width = picture.shape[1]
    # Инициализируем рандом ключом
    np.random.seed(key)
    #Использованные координаты
    before = []

    for i in range(N):
        # Генерируем координаты
        x, y = generate_unique_x_y(before, height, width)
        before.append((x, y))
        #Извлекаем
        if picture[x][y][blue_index] <= I(omega, picture, x, y):
            res_bites.append(1)
        else:
            res_bites.append(0)
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

#Метод для подсчета операции Лаплласа
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
    image = read_image("cat.jpg")
    # Чтение сообщения из файла message.txt
    lines = []
    with open("message.txt", 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line)
    text = ''.join(lines)
    # Преобразование сообщения в массив битов
    bites = to_bits(text)

    energy = 1
    skimage.io.imsave("before-blue-channel.jpg", extract_blue_channel(image))

    omega = 5
    key = 100
    # Получение заполненного стегокнтейнера
    encoded_image = hide_message(bites, image, energy, omega, key=key)
    skimage.io.imsave("after-blue-channel.jpg", extract_blue_channel(encoded_image))
    skimage.io.imsave("encoded-cat.jpg", encoded_image)

    # Получение соолбщения из заполненного стегокнтейнера
    decoded_bits_message = get_message_from_image(encoded_image, omega, len(bites), key=key)

    # Вывод строки из полученного массива битов сообщения
    print(from_bits(decoded_bits_message))

    with open("message-from-image.txt", 'w', encoding='utf8') as f:
        f.write(from_bits(decoded_bits_message))
    # подсчет метрик на основе незаполненного и заполненного контейнеров
    metrics(image, encoded_image)
