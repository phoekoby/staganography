import math
import numpy as np
import skimage.io

block_size = 8

# Получение из картинки только голубого канала
def extract_blue_channel(picture):
    blue_channel = picture[:, :, 2]
    blue_img = np.zeros(picture.shape)
    blue_img[:, :, 2] = blue_channel
    return blue_img


# Первый этап, сортировка по яркости, затем делим на две части и разбиваем на категори
def group_by_brightness(image):
    a = np.zeros((block_size ** 2, 3))
    n = 0
    # Составляем массив с координатами и значением яркости
    for i in range(block_size):
        for j in range(block_size):
            a[n][0] = i
            a[n][1] = j
            r, g, b = image[i, j, 0:3]
            a[n][2] = r * 0.299 + g * 0.587 + b * 0.114
            n += 1
    # Сортируем по яркости
    a = a[np.argsort(a[:, 2])]

    mask = np.zeros((block_size, block_size), dtype=np.int_)

    # Составляем маску
    for b in a[:(block_size ** 2) // 2]:
        i = math.floor(b[0])
        j = math.floor(b[1])
        mask[i][j] = 0

    for b in a[(block_size ** 2) // 2:]:
        i = math.floor(b[0])
        j = math.floor(b[1])
        mask[i][j] = 1
    return mask

# Генерация случайной маски (разбиение на группы А и B)
def mask():
    m = np.zeros((block_size, block_size), dtype=np.int_)
    for i in range(block_size):
        for j in range(block_size):
            m[i][j] = int(np.random.rand() * 2)
    return m

# Подсчет средней яркости группы и категории
def brightness(image, mask):
    l = 0
    n = np.sum(mask)
    for i in range(block_size):
        for j in range(block_size):
            r, g, b = image[i, j, 0:3]
            pix_br = r * 0.299 + g * 0.587 + b * 0.114
            if mask[i][j] == 1:
                l += pix_br
    l = l / n
    return l, n

# Увеличиваем яркость пикселей на маске
def add_brightness(image, mask, delta):
    for i in range(block_size):
        for j in range(block_size):
            if mask[i, j] == 0:
                continue
            r, g, b = image[i, j, 0:3]

            r += delta
            b += delta
            g += delta
            if r > 255:
                r = 255
            if g > 255:
                g = 255
            if b > 255:
                b = 255
            image[i, j, 0:3] = r, g, b
    return image

# Уменьшаем яркость пикселей на маске
def remove_brightness(image, mask, delta):
    for i in range(block_size):
        for j in range(block_size):
            if mask[i, j] == 0:
                continue
            r, g, b = image[i, j, 0:3]
            r -= delta
            g -= delta
            b -= delta
            if r < 0:
                r = 0
            if g < 0:
                g = 0
            if b < 0:
                b = 0
            image[i, j, 0:3] = r, g, b
    return image

# Метод скрытия сообщения
def hide_message(message_bits, img, l_b, key, delta):
    img = img.copy()
    # Инициализация рандом ключом
    np.random.seed(key)

    r_mask = mask()  # A = 0, B = 1
    m_i = 0

    w = img.shape[1]
    h = img.shape[0]

    w_b = math.floor(w / block_size)
    h_b = math.floor(h / block_size)
    # Идем по всем блокам в картинке
    for i in range(h_b):
        for j in range(w_b):
            if m_i >= l_b:
                break

            x = j * block_size
            y = i * block_size

            # Группируем по яркости
            group = group_by_brightness(img[y:y + block_size, x:x + block_size, :])

            # Накладываем маски группы и категории
            mask_1_a = 1 * np.logical_and(np.logical_not(r_mask), np.logical_not(group))
            mask_2_a = 1 * np.logical_and(np.logical_not(r_mask), group)
            mask_1_b = 1 * np.logical_and(r_mask, np.logical_not(group))
            mask_2_b = 1 * np.logical_and(r_mask, group)

            bit = message_bits[m_i]
            # Увлеичиваем или уменьшаем яркость в зависимости от значения бита
            if bit == 1:
                remove_brightness(img[y:y + block_size, x:x + block_size, :], mask_1_b, delta)
                add_brightness(img[y:y + block_size, x:x + block_size, :], mask_2_a, delta)
            else:
                remove_brightness(img[y:y + block_size, x:x + block_size, :], mask_1_a, delta)
                add_brightness(img[y:y + block_size, x:x + block_size, :], mask_2_b, delta)
            m_i += 1

        if m_i >= l_b:
            break
    return img


# Метод получения сообщения из картинки
def get_message_from_image(picture, l_b, key):
    picture = picture.copy()

    # Инициализация рандом ключом
    np.random.seed(key)

    r_mask = mask() # A = 0, B = 1

    w = picture.shape[1]
    h = picture.shape[0]

    w_b = math.floor(w / block_size)
    h_b = math.floor(h / block_size)

    m_i = 0
    res_bits = []

    # Идем по всем блокам
    for i in range(h_b):
        for j in range(w_b):
            if m_i >= l_b:
                break

            x = j * block_size
            y = i * block_size

            # Макса группы по яркости
            group = group_by_brightness(picture[y:y + block_size, x:x + block_size, :])

            # Накладываем маски группы и категории
            mask_1_a = 1 * np.logical_and(np.logical_not(r_mask), np.logical_not(group))
            mask_2_a = 1 * np.logical_and(np.logical_not(r_mask), group)
            mask_1_b = 1 * np.logical_and(r_mask, np.logical_not(group))
            mask_2_b = 1 * np.logical_and(r_mask, group)

            # Находим средние значения для каждой группы-категории блока
            l_1_a, n_1_a = brightness(picture[y:y + block_size, x:x + block_size, :], mask_1_a)
            l_2_a, n_2_a = brightness(picture[y:y + block_size, x:x + block_size, :], mask_2_a)
            l_1_b, n_1_b = brightness(picture[y:y + block_size, x:x + block_size, :], mask_1_b)
            l_2_b, n_2_b = brightness(picture[y:y + block_size, x:x + block_size, :], mask_2_b)

            bit = 0
            # Вытаскиваем бит
            if l_1_a - l_1_b < 0 and l_2_a - l_2_b < 0:
                bit = 0
            elif l_1_a - l_1_b > 0 and l_2_a - l_2_b > 0:
                bit = 1

            res_bits.append(bit)
            m_i += 1

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
    image = read_image("png-cat.png")
    # Чтение сообщения из файла message.txt
    lines = []
    with open("message.txt", 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line)
    text = ''.join(lines)
    # Преобразование сообщения в массив битов
    bites = to_bits(text)

    skimage.io.imsave("before-blue-channel.png", extract_blue_channel(image))

    #ключ
    key = 100

    # дельта l
    delta = 10
    # Получение заполненного стегокнтейнера
    encoded_image = hide_message(bites, image, len(bites), key=key, delta=delta)
    skimage.io.imsave("after-blue-channel.png", extract_blue_channel(encoded_image))
    skimage.io.imsave("encoded-cat-png.png", encoded_image)
    # Получение соолбщения из заполненного стегокнтейнера

    decoded_bits_message = get_message_from_image(encoded_image, len(bites), key)
    # Вывод строки из полученного массива битов сообщения
    print(from_bits(decoded_bits_message))

    with open("message-from-image.txt", 'w', encoding='utf8') as f:
        f.write(from_bits(decoded_bits_message))
    # подсчет метрик на основе незаполненного и заполненного контейнеров
    metrics(image, encoded_image)
