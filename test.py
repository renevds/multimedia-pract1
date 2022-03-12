import numpy as np

from gifencoder import calculate_psnr, make_grayscale_color_table


def simple_psnr_test():
    a = np.array([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
    b = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
    assert (calculate_psnr(a, b) == calculate_psnr(b, a))
    print(calculate_psnr(a, b))


def expected_value_psnr_test():
    res = []
    for i in range(100):
        img1 = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        res.append(calculate_psnr(img1, img2))
    return sum(res)/len(res)


if __name__ == '__main__':
    simple_psnr_test()
    print(f'should be around 7.75: {expected_value_psnr_test()}')
    a = make_grayscale_color_table(10)
    print(a)
    print(np.shape(a))
