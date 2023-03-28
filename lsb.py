import numpy as np
import skimage.io
import matplotlib.pyplot as plt


# Метод сокрытия сообщения в картинке
def hide_message(message_bits, picture):
    # Проверка вместимости стегоконтейнера, по сравнению с длинной сообщения
    if len(message_bits) > picture.shape[0] * picture.shape[1] * picture.shape[2]:
        print("I can not hide message")
        return picture
    # Получение размерности массива байтов изображения
    picture_shape = picture.shape
    # Зануление последнего бита каждого байта изображения и преобразование в одномерный массив
    picture = ((picture >> 1) << 1).reshape(-1)
    message_bits = np.asarray(message_bits)
    # получение размерности массива битов сообщения
    bits_length = message_bits.shape[0]
    # Битовое сложение каждого байта изображения с битом сообщения
    picture[:bits_length] = picture[:bits_length] | message_bits
    # Преобразование массива в исходную размерность
    return picture.reshape(picture_shape)


# Метод получения сообщения из картинки
def get_message_from_image(picture):
    # Преобразовние картинки в одномерный массив и битовое умножение с 0x01
    picture = picture.reshape(-1) & 0x01
    # Отсечение лишних битов и преобразование в массив [количество байт х 8]
    message = picture[:(picture.shape[0] // 8) * 8].reshape(-1, 8)
    # Получение байтов сообщения до первого нулевого байта и преобразования в одномерный массив
    return message[~np.all(message == 0, axis=1)].reshape(-1)


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
    with open("message.txt") as f:
        for line in f:
            lines.append(line)
    text = ''.join(lines)
    # Преобразование изображения в массив битов
    bites = to_bits(text)
    # Получение заполненного стегокнтейнера
    encoded_image = hide_message(bites, image)
    # Получение соолбщения из  заполненного стегокнтейнера
    decoded_bits_message = get_message_from_image(encoded_image)
    # Вывод строки из полученного массива битов сообщения
    print(from_bits(decoded_bits_message))
    # подсчет метрик на основе незаполненного и заполненного контейнеров
    metrics(image, encoded_image)


def graphic(image):
    x = []
    y = []
    for i in range(8, 4096, 8):
        nums = np.random.choice([0, 1], size=i, p=[.3, .7])
        encoded_image = hide_message(nums, image)
        decoded_nums = get_message_from_image(encoded_image)
        bolls = (nums == decoded_nums)
        ver = (np.size(bolls) - np.count_nonzero(bolls)) / np.size(bolls)
        x.append(i)
        y.append(ver)
    plt.plot(x, y)
    plt.show()
