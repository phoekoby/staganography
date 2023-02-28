import numpy as np
import skimage.io
import matplotlib.pyplot as plt


def hide_message(message_bits, picture):
    if len(message_bits) > picture.shape[0] * picture.shape[1] * picture.shape[2]:
        print("I can not hide message")
        return picture
    picture_shape = picture.shape
    picture = ((picture >> 1) << 1).reshape(-1)
    message_bits = np.asarray(message_bits)
    bits_length = message_bits.shape[0]
    picture[:bits_length] = picture[:bits_length] | message_bits
    return picture.reshape(picture_shape)


def get_message_from_image(picture):
    message = picture.reshape(-1) & 0x01
    message = message[:(message.shape[0] // 8) * 8].reshape(-1, 8)
    return message[~np.all(message == 0, axis=1)].reshape(-1)


def read_image(path):
    image = skimage.io.imread(path)
    return image


def to_bits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def from_bits(bits):
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


def metrics(empty, full):
    max_d = np.max(np.abs(empty.astype(int) - full.astype(int)))
    print(f"Максимальное абсолютное отклонение: %d" % max_d)
    m_NMSE = np.sum((empty - full) * (empty - full)) / np.sum((empty * empty))
    print(f"Нормированное среднее квадратичное отклонение: %f" % m_NMSE)
    m_SNR = 1 / m_NMSE
    print(f"Отношение сигнал-шум: %f" % m_SNR)
    H = empty.shape[0]
    W = empty.shape[1]
    m_PSNR = W * H * ((np.max(empty) ** 2) / np.sum((empty - full) * (empty - full)))
    print(f"Пиковое отношение сигнал-шум : %f" % m_PSNR)


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


if __name__ == "__main__":
    image = read_image("cat.jpg")
    lines = []
    with open("message.txt") as f:
        for line in f:
            lines.append(line)
    text = ''.join(lines)
    bites = to_bits(text)
    encoded_image = hide_message(bites, image)
    decoded_bits_message = get_message_from_image(encoded_image)
    print(from_bits(decoded_bits_message))
    metrics(image, encoded_image)
    graphic(image)
