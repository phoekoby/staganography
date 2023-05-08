from PIL import Image
import random
import numpy as np


def encode_image(src, str, res, delta):
    img = Image.open(src, 'r')
    array = np.array(list(img.getdata())) # Получение массива пикселей изображения.
    width, height = img.size
    count_pixels = array.size // 3 # Кол-во пикселей в изображении.
    str += "#s40g0"
    bin_str = ''.join([format(ord(i), "08b") for i in str])
    pixels = len(bin_str)
    new_pixels = []
    group = ''
    if pixels > count_pixels:
        print("Длина сообщения больше доступного количества пикселей")
    else:
        index = 0
        for y in range(0, height, 8):
            for x in range(0, width, 8):
                if index < pixels:
                    temp_new_pixels = []
                    for y1 in range(y, y + 8):
                        for x1 in range(x, x + 8):
                            end_index = width * y1 + x1
                            temp_new_pixels.append(array[end_index])
                    new_pixels.append(temp_new_pixels)
                    print("POST")
                    print(sort_to_bright(temp_new_pixels))
                    group += sort_to_bright(temp_new_pixels)
                    index += 1
        mask = set_mask(pixels, group)
        for i in range(pixels):
            if bin_str[i] == '1':
                numMask = 'b'
                numDelta = -delta
                for k in range(1, 3):
                    new_pixels[i] = change_bright(new_pixels[i], mask, group, numMask, k, numDelta, i)
                    numMask = 'a'
                    numDelta = delta
            else:
                numMask = 'a'
                numDelta = -delta
                for k in range(1, 3):
                    new_pixels[i] = change_bright(new_pixels[i], mask, group, numMask, k, numDelta, i)
                    numMask = 'b'
                    numDelta = delta
        index1 = 0
        for y in range(0, height, 8):
            for x in range(0, width, 8):
                if index1 < pixels:
                    constY = 0
                    for y1 in range(y, y + 8):
                        constX = 0
                        for x1 in range(x, x + 8):
                            array[width * y1 + x1] = new_pixels[index1][8 * constY + constX]
                            constX += 1
                        constY += 1
                    index1 += 1
        array = array.reshape(height, width, 3)
        enc_img = Image.fromarray(array.astype('uint8'), "RGB")
        enc_img.save(res)
    return mask

def change_bright(array, mask, group, numberMask, number, delta, n):
    for i in range(64):
        if mask[n*64 + i] == numberMask and group[n*64 + i] == str(number):
            array[i][0] += delta
            array[i][1] += delta
            array[i][2] += delta
    return array

def decode_image(src, mask):
    img = Image.open(src, 'r')
    width, height = img.size
    array = np.array(list(img.getdata()))
    check = True
    new_pixels1 = []
    bits = ''
    index = 0
    group1 = ''
    resString = ''
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            if "#s40g0" in resString:
                check = False
            if check is True:
                new_pixels = []
                for y1 in range(y, y + 8):
                    for x1 in range(x, x + 8):
                        end_index = width * y1 + x1
                        new_pixels.append(array[end_index])
                new_pixels1.append(new_pixels)
                print("GET")
                print(sort_to_bright(new_pixels))
                group1 += sort_to_bright(new_pixels)
                la1res, la2res, lb1res, lb2res = calculate_avg_values(mask, group1, new_pixels1[index], index)
                index += 1
                if la1res < lb1res and la2res < lb2res:
                    bits += '0'
                elif la1res > lb1res and la2res > lb2res:
                    bits += '1'
                bits1 = [bits[i:i + 8] for i in range(0, len(bits), 8)]
                final_message = ""
                for i in range(0, len(bits1)):
                    final_message += chr(int(bits1[i], 2))
                resString = final_message
    return print('Закодированное сообщение: ', resString[:-6])

def calculate_avg_values(mask, group, array, n):
    la1, la2, lb1, lb2 = 0, 0, 0, 0
    nb1, nb2, na1, na2 = 0, 0, 0, 0
    for i in range(64):
        R = array[i][0]
        G = array[i][1]
        B = array[i][2]
        L = int(0.299 * R + 0.587 * G + 0.114 * B)
        if mask[n*64 + i] == 'a' and group[n*64 + i] == '1':
            la1 += L
            na1 += 1
        elif mask[n*64 + i] == 'a' and group[n*64 + i] == '2':
            la2 += L
            na2 += 1
        elif mask[n*64 + i] == 'b' and group[n*64 + i] == '1':
            lb1 += L
            nb1 += 1
        elif mask[n*64 + i] == 'b' and group[n*64 + i] == '2':
            lb2 += L
            nb2 += 1

    la1res = la1 / na1
    la2res = la2 / na2
    lb1res = lb1 / nb1
    lb2res = lb2 / nb2
    return la1res, la2res, lb1res, lb2res


def set_mask(pixels, group):
    resultMask = ''
    for i in range(pixels):
        let_a1, let_b1, let_a2, let_b2 = 0, 0, 0, 0
        rand_string = ''
        for j in range(64):
            if group[i * 64 + j] == '1':
                if let_a1 >= 16:
                    rand_string += ''.join('b')
                elif let_b1 >= 16:
                    rand_string += ''.join('a')
                else:
                    rand_string += ''.join(random.choice('ab'))
                if rand_string[j] == 'a':
                    let_a1 += 1
                else:
                    let_b1 += 1
            elif group[i * 64 + j] == '2':
                if let_a2 >= 16:
                    rand_string += ''.join('b')
                elif let_b2 >= 16:
                    rand_string += ''.join('a')
                else:
                    rand_string += ''.join(random.choice('ab'))
                if rand_string[j] == 'a':
                    let_a2 += 1
                else:
                    let_b2 += 1
        resultMask += rand_string
    return resultMask

def sort_to_bright(new_pixels_tuple):
    new_pixels = np.array(new_pixels_tuple).tolist()
    for i in range(64):
        new_pixels[i].append(i)
    left = 0
    right = len(new_pixels) - 1
    while left <= right:
        for i in range(left, right, +1):
            R = new_pixels[i][0]
            G = new_pixels[i][1]
            B = new_pixels[i][2]
            R1 = new_pixels[i+1][0]
            G1 = new_pixels[i+1][1]
            B1 = new_pixels[i+1][2]
            L = int(0.299 * R + 0.587 * G + 0.114 * B)
            L1 = int(0.299 * R1 + 0.587 * G1 + 0.114 * B1)
            if L > L1:
                new_pixels[i], new_pixels[i+1] = new_pixels[i+1], new_pixels[i]
        right -= 1
        for i in range(right, left, -1):
            R = new_pixels[i][0]
            G = new_pixels[i][1]
            B = new_pixels[i][2]
            R1 = new_pixels[i - 1][0]
            G1 = new_pixels[i - 1][1]
            B1 = new_pixels[i - 1][2]
            L = int(0.299 * R + 0.587 * G + 0.114 * B)
            L1 = int(0.299 * R1 + 0.587 * G1 + 0.114 * B1)
            if L1 > L:
                new_pixels[i], new_pixels[i-1] = new_pixels[i-1], new_pixels[i]
        left += 1

    array = np.array(np.zeros(64)).tolist()

    for i in range(64):
        if i < 32:
            array[new_pixels[i][3]] = 1
        elif i >= 32:
            array[new_pixels[i][3]] = 2
    return " ".join(map(str, array)).replace(' ', '')


def metrics_evaluation(in_image, en_image):
    img = Image.open(in_image)
    enc_img = Image.open(en_image)

    arr = np.array(list(img.getdata()), dtype=np.int64)  # Получение массива пикселей исходного контейнера.
    enc_arr = np.array(list(enc_img.getdata()), dtype=np.int64)  # Получение массива пикселей измененного контейнера.

    max_d = np.amax(abs(arr - enc_arr))
    print(f"Максимальное абсолютное отклонение: %d" % max_d)

    q_nmse = str(np.sum(abs(arr - enc_arr) * abs(arr - enc_arr)) / np.sum(arr * arr))
    print("Нормированное среднее квадратичное отклонение: ", f"{float(q_nmse):.{int(q_nmse[-2:]) + 2}f}")

    width, height = img.size

    q_uqi = np.sum(4 * (1 / (width * height) * np.sum((arr - np.mean(arr))) * (enc_arr - np.mean(enc_arr))) * np.mean(
        arr) * np.mean(enc_arr) \
                   / ((np.var(arr) ** 2 + np.var(enc_arr) ** 2) * (np.mean(arr) ** 2 + np.mean(enc_arr) ** 2)))
    print(f"Универсальный индекс качества: %s" % str(q_uqi))
    return

# Преобразование строки символов в массив бит.
def string_to_bin(message):
    chain = list()
    for char in message:
        binary_string = '{0:08b}'.format(ord(char))
        for value in binary_string:
            chain.append(int(value))
    return chain

# Преобразование массива бит в строку символов.
def bin_to_string(chain):
    message_string = ''
    chain_pointer = 0
    while chain_pointer + 8 <= len(chain):
        binary_string = ''.join(str(x) for x in chain[chain_pointer:chain_pointer + 8])
        message_string += chr(int(binary_string, 2))
        chain_pointer += 8
    return message_string

# Чтение встраиваемого сообщения.
def load_text(text_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def main(im_path, text_path, out_im_path):
    input_message = load_text(text_path)

    # Встраивание сообщения в стегоконтейнер.
    e_image = encode_image(im_path, input_message, out_im_path, 5)

    # Получение сообщения из заполненного стегокнтейнера.
    decode_image(out_im_path, e_image)

    # Подсчет метрик.
    metrics_evaluation(im_path, out_im_path)

    return

if __name__ == '__main__':
    main("png-cat.png", "message.txt", "new.png")