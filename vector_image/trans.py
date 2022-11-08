import cv2 as cv
import numpy as np
from numpy import ndarray
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import threading

threadLock = threading.Lock()


class DrawDFSThread(threading.Thread):

    def __init__(self, threadID, src, keys, serials):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.src = src
        self.keys = keys
        self.serials = serials

    def run(self):
        print("Starting " + self.name)
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定
        # 否则超时后将返回False
        threadLock.acquire()
        show_dfs_result(self.src, self.keys, self.serials)
        # 释放锁
        threadLock.release()


class DrawREThread(threading.Thread):

    def __init__(self, threadID, src, coefficients_list, step):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.src = src
        self.coefficients_list = coefficients_list
        self.step = step

    def run(self):
        print("Starting " + self.name)
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定
        # 否则超时后将返回False
        threadLock.acquire()
        show_regularized_curve(self.src, self.coefficients_list, self.step)
        # 释放锁
        threadLock.release()


def array_reset(array):
    a = array[:, 0: len(array[0]) - 2: 3]
    return a


def depth_first_search(src, keys, serial):
    rows = src.shape[0]
    cols = src.shape[1]
    total = rows * cols

    index = 0
    processed = [False for i in range(total)]
    orients = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

    count = 0
    while index < total:
        stack = []
        i = int(index / cols)
        j = index % cols
        if processed[index]:
            index += 1
            continue
        elif src[i][j] == 0:
            processed[index] = True
            index += 1
            continue

        stack.append((i, j))
        processed[index] = True
        keys.append(count)

        last_orient = (0, 0)

        def is_corner(x, y):
            # print("dot: ", x[0] * y[0] + x[1] * y[1])
            return (x[0] * y[0] + x[1] * y[1]) < 0

        while len(stack) != 0:
            (i, j) = stack.pop(-1)
            serial.append((i, j))
            # now_orient = (0, 0)
            is_end = True
            for orient in orients:
                now = (i + orient[0], j + orient[1])
                now_idx = now[0] * cols + now[1]
                if now[0] >= rows or now[0] < 0:
                    continue
                if now[1] >= cols or now[1] < 0:
                    continue
                if src[now[0]][now[1]] == 0:
                    continue
                if processed[now_idx]:
                    continue
                stack.append(now)
                processed[now_idx] = True
                # now_orient = orient
                is_end = False
            count += 1
            if is_end:
                keys.append(count)
                # print("{}th curve".format(len(keys) - 1))
            # elif is_corner(last_orient, now_orient):
            #     keys.append(count)
            # print("{}th curve".format(len(keys) - 1))
            # last_orient = now_orient
        index += 1


def regularize_curves(keys, serial, every_length):
    get_length = lambda point1, point2: np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def least_square(A, MX) -> list:
        try:
            return ((A.T * A).I * (A.T * MX)).tolist()
        except np.linalg.LinAlgError:
            print('Error: LinAlgError Occurs!')
            print(A.T * A)
            exit(-1)

    coefficients_serial = []
    # print('serial length: ', len(serial))
    for i in range(len(keys) - 1):
        P = []
        S = []
        X = []
        Y = []
        coefficients_in_one_segment = []
        current_length = 0
        # print('keys[i]: ', keys[i])
        last_pixel = [0, 0]
        last_tangent_x = 0
        last_tangent_y = 0
        d_x = 0
        d_y = 0
        c_x = 0
        c_y = 0
        count = 0
        is_first = True
        is_new_segment = True
        if keys[i + 1] - keys[i] <= 3:
            continue

        for (x, y) in serial[keys[i]: keys[i + 1]]:
            if is_new_segment:
                last_pixel[0] = x
                last_pixel[1] = y
                is_new_segment = False
            current_length += get_length((x, y), last_pixel)
            if is_first:  # if this is the first segment in a curve
                if current_length == 0:  # if this is the first point in a segment
                    d_x = x  # d = x_0
                    d_y = y  # d = y_0
                S.append(current_length)
                # S.append([current_length ** 3, current_length ** 2, current_length])
                X.append([x])
                Y.append([y])
                count += 1
                if current_length >= every_length or count == keys[i + 1] - keys[i]:
                    if len(S) <= 3:
                        break
                    xn = X[-1][0]
                    yn = Y[-1][0]
                    for index, s in enumerate(S):
                        X[index][0] -= (xn - d_x) / current_length * s + d_x
                        Y[index][0] -= (yn - d_y) / current_length * s + d_y
                        P.append([s ** 3 - s * (current_length ** 2), s ** 2 - s * current_length])
                    m_p = np.matrix(P)
                    m_x = np.matrix(X)
                    m_y = np.matrix(Y)
                    coefficients_x = least_square(m_p, m_x)
                    c_x = (xn - d_x) / current_length - coefficients_x[0][0] * current_length ** 2 \
                          - coefficients_x[1][0] * current_length
                    coefficients_x.append([c_x])
                    coefficients_x.append([d_x])
                    coefficients_y = least_square(m_p, m_y)
                    c_y = (yn - d_y) / current_length - coefficients_y[0][0] * current_length ** 2 \
                          - coefficients_y[1][0] * current_length
                    coefficients_y.append([c_y])
                    coefficients_y.append([d_y])
                    coefficients = (coefficients_x, coefficients_y, current_length)
                    coefficients_in_one_segment.append(coefficients)
                    last_tangent_x = 3 * coefficients_x[0][0] * current_length ** 2 \
                                     + 2 * coefficients_x[1][0] * current_length \
                                     + coefficients_x[2][0]
                    last_tangent_y = 3 * coefficients_y[0][0] * current_length ** 2 \
                                     + 2 * coefficients_y[1][0] * current_length \
                                     + coefficients_y[2][0]
                    is_first = False
                    P = []
                    S = []
                    X = []
                    Y = []
                    current_length = 0
                    is_new_segment = True
                    continue

            else:  # if this is not the first segment in a curve
                if current_length == 0:  # if this is the first point in a segment
                    c_x = last_tangent_x  # c = x_0'
                    c_y = last_tangent_y  # c = x_0'
                    d_x = x  # d = x_0
                    d_y = y  # d = y_0
                S.append(current_length)
                # S.append([current_length ** 3, current_length ** 2])
                X.append([x])
                Y.append([y])
                count += 1
                if current_length >= every_length or count == keys[i + 1] - keys[i]:
                    if len(S) < 3:
                        break
                    xn = X[-1][0]
                    yn = Y[-1][0]
                    for index, s in enumerate(S):
                        X[index][0] -= (X[-1][0] - c_x * current_length - d_x) / (current_length ** 2) * (s ** 2) \
                                       + c_x * s + d_x
                        Y[index][0] -= (Y[-1][0] - c_y * current_length - d_y) / (current_length ** 2) * (s ** 2) \
                                       + c_y * s + d_y
                        P.append([s ** 3 - current_length * s ** 2])
                    m_p = np.matrix(P)
                    m_x = np.matrix(X)
                    m_y = np.matrix(Y)
                    coefficients_x = least_square(m_p, m_x)
                    b_x = (xn - c_x * current_length - d_x) / (current_length ** 2) \
                          - coefficients_x[0][0] * current_length
                    coefficients_x.append([b_x])
                    coefficients_x.append([c_x])
                    coefficients_x.append([d_x])
                    coefficients_y = least_square(m_p, m_y)
                    b_y = (yn - c_y * current_length - d_y) / (current_length ** 2) \
                          - coefficients_y[0][0] * current_length
                    coefficients_y.append([b_y])
                    coefficients_y.append([c_y])
                    coefficients_y.append([d_y])
                    coefficients = (coefficients_x, coefficients_y, current_length)
                    coefficients_in_one_segment.append(coefficients)
                    last_tangent_x = 3 * coefficients_x[0][0] * current_length ** 2 \
                                     + 2 * coefficients_x[1][0] * current_length \
                                     + c_x
                    last_tangent_y = 3 * coefficients_y[0][0] * current_length ** 2 \
                                     + 2 * coefficients_y[1][0] * current_length \
                                     + c_y
                    P = []
                    S = []
                    X = []
                    Y = []
                    current_length = 0
                    is_new_segment = True
                    continue

            last_pixel[0] = x
            last_pixel[1] = y
        coefficients_serial.append(coefficients_in_one_segment)
    # print('coefficient lens: ', len(coefficients_serial))
    return coefficients_serial


def show_dfs_result(src, keys, serial):
    row = src.shape[0]
    col = src.shape[1]

    print("There are {} clusters of curves.".format(len(keys) - 1))

    curve = np.zeros((row, col), dtype=np.float32)
    for i in range(len(keys) - 1):
        count = 2
        if keys[i + 1] - keys[i] <= 3:
            continue
        for (x, y) in serial[keys[i]: keys[i + 1]]:
            if count == 2:
                count = 0
                curve[x][y] = 1
                cv.imshow("curve".format(i), curve)
                cv.waitKey(3)
            count += 1
        cv.waitKey(500)
    cv.waitKey(0)


def show_regularized_curve(src, coefficients_list, step=0):
    row = int(src.shape[0])
    col = int(src.shape[1])

    def func(coefficients, ix):
        result = 0
        length = len(coefficients)
        for index, coefficient in enumerate(coefficients):
            result += coefficient * (ix ** (length - index - 1))
        return int(result)

    curve = np.zeros((row, col), dtype=np.uint8)
    # print('curve size: ', curve.shape)
    # print('list len: ', len(coefficients_list))
    for index, curve_coefficients in enumerate(coefficients_list):
        for segment_coefficients in curve_coefficients:
            s = 0
            coefficients_x = [coefficient[0] for coefficient in segment_coefficients[0]]
            coefficients_y = [coefficient[0] for coefficient in segment_coefficients[1]]
            max_length = segment_coefficients[2]
            while s < max_length:
                x = func(coefficients_x, s)
                y = func(coefficients_y, s)
                if x >= row or y >= col or x < 0 or y < 0:
                    continue
                curve[x][y] = 255
                if step == 0:
                    s += 1
                    continue
                # cv.imshow('regularized_curve', curve)
                # cv.waitKey(3)
                s += step
            # cv.waitKey(3)
    # cv.imshow('regularized_curve', curve)
    # cv.waitKey(0)
    return curve


def color_distribution(src: ndarray, color_type='rgb'):
    if color_type == 'rgb':
        color_r = np.array([0 for i in range(256)])
        color_g = np.array([0 for i in range(256)])
        color_b = np.array([0 for i in range(256)])
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                color_r[src[i][j][0]] += 1
                color_g[src[i][j][1]] += 1
                color_b[src[i][j][2]] += 1
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(0, 256), color_r, color='r', label='r distribution')
        plt.xlim([-10, 255])
        plt.xlabel('color')
        plt.ylabel('times')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(np.arange(0, 256), color_g, color='g', label='g distribution')
        plt.xlim([-10, 255])
        plt.xlabel('color')
        plt.ylabel('times')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(np.arange(0, 256), color_b, color='b', label='b distribution')
        plt.xlim([-10, 255])
        plt.xlabel('color')
        plt.ylabel('times')
        plt.legend()

        plt.show()

    if color_type == 'gray':
        gray_scale = np.array([0 for i in range(256)])
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                gray_scale[src[i][j]] += 1

        plt.plot(np.arange(0, 256), gray_scale, color='gray', label='gray distribution')
        plt.xlim([-10, 255])
        plt.xlabel('color')
        plt.ylabel('times')
        plt.legend()

        plt.show()


def color_means(src, clusters_num):
    row = src.shape[0]
    col = src.shape[1]
    img_means = src.copy()
    img_means_vec = img_means.reshape((row * col, 3))
    kmeans = KMeans(n_clusters=clusters_num, init='k-means++', max_iter=100, n_init=5)
    kmeans.fit(img_means_vec)

    for i in range(img_means.shape[0]):
        for j in range(img_means.shape[1]):
            img_means[i][j] = kmeans.cluster_centers_[kmeans.labels_[i * col + j]]

    dst = img_means
    return dst


def gaussian_blur(src, kernel_size, mean):
    dst = cv.GaussianBlur(src, (kernel_size, kernel_size), mean)
    return dst


def mean_shift(src, sp, sr):
    dst = cv.pyrMeanShiftFiltering(src, sp, sr)
    return dst


def dilate_and_erode(src, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dst = cv.morphologyEx(src, cv.MORPH_OPEN, kernel, iterations=iterations)
    dst = cv.morphologyEx(dst, cv.MORPH_CLOSE, kernel, iterations=iterations)
    return dst


def resize_image(src):
    min_size = min(src.shape[0], src.shape[1])
    max_size = 297 * 4
    scale = 840 / min_size
    dst = cv.resize(src, (0, 0), fx=scale, fy=scale)
    row = dst.shape[0]
    col = dst.shape[1]
    print('dst size: ', dst.shape)
    if dst.shape[1] == 840:
        mid = np.zeros((col, row, dst.shape[2]), dtype=np.uint8)
        for i in range(dst.shape[2]):
            mid[:, :, i] = dst[:, :, i].T
        dst = mid
        dst = np.flip(dst, 0)
    if dst.shape[1] > max_size:
        left_offset = int((dst.shape[1] - max_size) / 2)
        right_offset = dst.shape[1] - int((dst.shape[1] - max_size + 1) / 2)
        dst = dst[:, left_offset: right_offset]
    else:
        left_offset = int((max_size - dst.shape[1]) / 2)
        right_offset = max_size - int((max_size - dst.shape[1] + 1) / 2)
    print('img size: ', dst.shape)
    return dst


def preprocess(img_origin, cluster_nums=4, gaussian_blur_parse=(False, 1, 0),
               mean_shift_parse=(False, 10, 100), dilate_and_erode_parse=(False, 1, 1)):
    if mean_shift_parse[0]:
        img_origin = mean_shift(img_origin, mean_shift_parse[1], mean_shift_parse[2])
    if dilate_and_erode_parse[0]:
        img_origin = dilate_and_erode(img_origin, dilate_and_erode_parse[1], dilate_and_erode_parse[2])
    dst = color_means(img_origin, cluster_nums)
    if gaussian_blur_parse[0]:
        dst = gaussian_blur(dst, gaussian_blur_parse[1], gaussian_blur_parse[2])
    return dst


def contour(img_origin, dilate_and_erode_parse=(False, 1, 1)):
    row = img_origin.shape[0]
    col = img_origin.shape[1]
    gray = cv.cvtColor(img_origin, cv.COLOR_RGB2GRAY)
    edged = cv.Laplacian(gray, 0, 3)
    for i in range(row):
        for j in range(col):
            if edged[i][j] > 0:
                edged[i][j] = 255
    if dilate_and_erode_parse[0]:
        edged = dilate_and_erode(edged, dilate_and_erode_parse[1], dilate_and_erode_parse[2])
    contours, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    img_result = np.zeros((row, col), dtype=np.uint8)
    cv.drawContours(img_result, contours, -1, 255)
    print('result size: ', img_result.shape)
    return img_result


def generate(img_contours, seg_length=20, step=2):
    serial = []
    keys = []
    depth_first_search(img_contours, keys, serial)
    # show_dfs_result(img_processed, keys, serial)

    coefficients_list = regularize_curves(keys, serial, seg_length)
    # show_regularized_curve(img_processed, coefficients_list, 10)

    dst = show_regularized_curve(img_contours, coefficients_list, step)

    return dst, coefficients_list


def bezier(A: tuple, B: tuple, C: tuple, x: list, y: list):
    D = [((B[0] - A[0]) * float(i) / 1000 + A[0], (B[1] - A[1]) * float(i) / 1000 + A[1]) for i in range(1001)]
    E = [((C[0] - B[0]) * float(i) / 1000 + B[0], (C[1] - B[1]) * float(i) / 1000 + B[1]) for i in range(1001)]
    print(D[0], D[-1])
    print(E[0], E[-1])
    x.append([(E[i][0] - D[i][0]) * float(i) / 1000 + D[i][0] for i in range(1001)])
    y.append([(E[i][1] - D[i][1]) * float(i) / 1000 + D[i][1] for i in range(1001)])

    plt.scatter([A[0]], [A[1]], s=16, c='b')
    plt.scatter([B[0]], [B[1]], s=20, c='r')
    plt.scatter([C[0]], [C[1]], s=16, c='b')


def draw_bezier(x: list, y: list):
    print(x[0], ' ', x[-1])
    print(y[0], ' ', y[-1])

    for i in range(len(x)):
        plt.scatter(x[i], y[i], s=1, c='gray')

    plt.show()


def main():
    # file = 'F:\\Self_Study\\6th_sem\\e_m_practice\\files\\izumi2.png'
    file = input('enter a filename: ')
    path = os.path.join(os.path.dirname(__file__), os.path.pardir)
    file = path + '\\imgs\\' + file
    print('absolute path: {0}'.format(file))
    img_origin = cv.imread(file, cv.IMREAD_COLOR)
    img_origin = contour(img_origin)
    generate(img_origin)


if __name__ == '__main__':
    main()
