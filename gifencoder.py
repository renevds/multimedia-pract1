import argparse
from operator import itemgetter

import numpy as np
from PIL import Image
from sklearn import cluster

import lzw

MAX = 255


def calculate_psnr(img_A, img_B):
    img_A = img_A.astype(float)
    img_B = img_B.astype(float)
    shape = np.shape(img_A)
    MSE = ((img_A - img_B) ** 2.0).sum(axis=1).sum(axis=0) / (shape[0] * shape[1])
    PSNR = 10.0 * np.log10((MAX ** 2.0 / MSE).sum() / 3.0)
    return PSNR


def make_random_color_tabel(color_table_size):
    """
    Makes a random color table with shape (color_table_size, 3) with
    dtype np.uint8. Consequently, color values are in range 0 - 255.
    """
    return np.random.randint(0, 256, (color_table_size, 3), dtype=np.uint8)


def make_grayscale_color_table(color_table_size):
    """
    Makes an evenly-spaced color table with shape (color_table_size, 3),
    with dtype np.uint8 filled with grayscale values ranging from 0 to 255.
    """
    return np.rint(
        np.repeat(np.reshape((np.arange(color_table_size) / (color_table_size - 1)) * 255, (color_table_size, 1)),
                  repeats=3, axis=1)).astype(np.uint8)


def make_random_sample_color_table(color_table_size, img):
    """
    Takes a random sample of colors from the supplied image `img` to construct
    a color table of shape (color_table_size, 3). Shape of `img` is
    (height, width, 3).
    """
    colors = img.reshape(-1, 3)
    rows = colors.shape[0]
    random_indices = np.random.choice(rows, size=color_table_size, replace=False)
    return colors[random_indices, :]


def median_help(bucket):
    if len(bucket) == 1:
        return [bucket, bucket]
    var_index = np.argmax(np.var(bucket, axis=0))
    sorted_bucket = sorted(bucket, key=itemgetter(var_index))
    mid = int(round(len(sorted_bucket)/2.0))
    return [sorted_bucket[:mid], sorted_bucket[mid:]]


def make_median_cut_color_table(color_table_size, img):
    """
    Makes a color table with shape (color_table_size, 3), based on
    the non-recursive median cut algorithm. Shape of `img` is
    (height, width, 3).
    """
    buckets = [[i for i in np.unique(img.reshape(-1, 3), axis=0)]]

    while len(buckets) < color_table_size:
        new_buckets = []
        for i in buckets:
            new_buckets += median_help(i)
        buckets = new_buckets

    size = int(round(len(buckets[0]) / 2))
    ret = []
    for i in buckets:
        ret.append(np.average(i, axis=0))

    return np.array(ret).astype(np.uint8)


def make_kmeans_color_table(color_table_size, img):
    """
    Makes a color table with shape (color_table_size, 3) and dtype np.uint8,
    based on the k-means clustering algorithm. Shape of `img` is (height, width, 3).
    """
    colors = img.reshape(-1, 3)
    kmeans = cluster.MiniBatchKMeans(color_table_size, n_init=4)
    res = kmeans.fit(colors).cluster_centers_
    print(res)
    return np.rint(res).astype(np.uint8)


def find_nearest_color_index(color_table, rgb_vec):
    """
    Finds for each element in `rgb_vec` the nearest (based on
    Euclidean distance) color present in the `color_table`,
    and yields its index.

    Parameters:
     - `color_table` has shape (N, D)
     - `rgb_vec` has shape (M, D)
    Where D is the number of dimensions of the color, which
    is typically 3.
    """
    distances = np.square(
        color_table.astype(np.float32) -
        np.expand_dims(rgb_vec.astype(np.float32), axis=-2))
    distances = np.sum(distances, axis=-1)
    return np.argmin(distances, axis=-1)


def transform_image_to_indices_no_dithering(img, color_table):
    """
    Returns the matrix of indices represeting colors in the supplied color table,
    obtained from finding the closest color for each pixel in said color table.
    Parameters:
        - `img`, shape (height, width, 3)
        - `color_table`, shape (N, 3)
    Returns:
        np.array with shape (width, height), and dtype np.uint8.
    """
    color_table_indices = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for row in range(img.shape[0]):
        color_table_indices[row] = find_nearest_color_index(color_table, img[row])
    return color_table_indices


def transform_image_to_indices_diffusion_dithering(img, color_table, dither_matrix, anchor_col):
    """
    Transforms a full-color image to the grid of indices where every full-color pixel is mapped
    onto one index. This process applies error diffusion using the supplied `dither_matrix`.
    The algorithm diffuses errors either:
        - to the right on the same scanline,
        - or downward (both left and right are possible).
    The `anchor_col` defines which column of the `dither_matrix` contains the anchor point.
    """
    rows = img.shape[0]
    cols = img.shape[1]
    color_table_indices = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            img[i][j] = img[i][j].astype(float)

    for i in range(rows):
        for j in range(cols):
            new_index = find_nearest_color_index(color_table, img[i, j])
            new = color_table[new_index].astype(float)
            old = img[i, j]
            color_table_indices[i, j] = new_index
            error = old - new
            img[i, j] = new
            for y in range(len(dither_matrix)):
                for x in range(len(dither_matrix[0])):
                    dy = j - anchor_col + x
                    dx = i + y
                    if 0 <= dx < rows and 0 <= dy < cols:
                        img[dx, dy] = (error * dither_matrix[y][x]) + img[dx, dy]
    return color_table_indices


def transform_image_to_indices_dithering_1(img, color_table):
    """
    Returns the matrix of indices represeting colors in the supplied color table.
    Chosen colors are influenced by dithering based on the Floyd-Steinberg
    diffusion algorithm.
    Parameters:
        - `img`, shape (height, width, 3)
        - `color_table`, shape (N, 3)
    Returns:
        np.array with shape (width, height), and dtype np.uint8.
    """
    kernel = np.array([[0, 0, 7],
                       [3, 5, 1]], dtype=np.float32) / 16
    return transform_image_to_indices_diffusion_dithering(img, color_table,
                                                          dither_matrix=kernel, anchor_col=1)


def transform_image_to_indices_dithering_2(img, color_table):
    """
    Returns the matrix of indices represeting colors in the supplied color table.
    Chosen colors are influenced by dithering based on the "Minimized Averaged Error"
    diffusion algorithm, by Bell Labs.
    Parameters:
        - `img`, shape (height, width, 3)
        - `color_table`, shape (N, 3)
    Returns:
        np.array with shape (width, height), and dtype np.uint8.
    """
    kernel = np.array([[0, 0, 0, 7, 5],
                       [3, 5, 7, 5, 3],
                       [1, 3, 5, 3, 1]], dtype=np.float32) / 48
    return transform_image_to_indices_diffusion_dithering(img, color_table,
                                                          dither_matrix=kernel, anchor_col=2)


def encode_gif(indices, color_table, output_path):
    assert len(indices.shape) == 2, 'indices must be a 2D array representing one index per pixel of the input image.'
    assert indices.dtype == np.uint8, 'indices must be of type uint8.'
    color_table_size = color_table.shape[0]
    assert (color_table_size & (color_table_size - 1)) == 0, 'color table size must be a power of two.'
    color_table_bits = int(np.log2(color_table_size) + 0.1)
    with open(output_path, 'wb') as w:
        # "GIF89a" in Hex
        w.write(bytes([0x47, 0x49, 0x46, 0x38, 0x39, 0x61]))

        # width and height in unsigned 2 byte (16 bit) little-endian
        width_bytes = (indices.shape[1]).to_bytes(2, byteorder='little')
        height_bytes = (indices.shape[0]).to_bytes(2, byteorder='little')
        w.write(width_bytes)
        w.write(height_bytes)

        # GCT follows for 256 colors with resolution 3 x 8 bits/primary;
        # the lowest 3 bits represent the bit depth minus 1, the highest
        # true bit means that the GCT is present
        w.write(bytes([0xf0 + color_table_bits - 1]))

        # Background color #0
        w.write(bytes([0x00]))
        # Default pixel aspect ratio
        w.write(bytes([0x00]))

        # Global color table (GCT)
        for c in range(color_table_size):
            r, g, b = color_table[c]
            w.write(bytes([r & 0xff, g & 0xff, b & 0xff]))

        # Graphic Control Extension (comment fields precede this in most files)
        w.write(bytes([0x21, 0xf9, 0x03, 0x00, 0x00, 0x00, 0x00]))

        # Image Descriptor
        w.write(bytes([0x2c]))
        w.write(bytes([0x00, 0x00, 0x00, 0x00]))  # NW corner position of image in logical screen
        w.write(width_bytes)
        w.write(height_bytes)

        w.write(bytes([0x00]))  # no local color table
        lzw_min = max(2, color_table_bits)
        max_code_size = 10

        # start of image - LZW minium
        w.write(lzw_min.to_bytes(1, byteorder='little'))

        color_table_indices = ''.join([chr(x) for x in indices.flatten()])
        compressed_indices = lzw.compress(color_table_indices, lzw_min, max_code_size)

        for i, byte in enumerate(compressed_indices):
            if i % 255 == 0:
                # Write length of coded stream in bytes (subblock can maximum be 255 long)
                w.write((min(255, len(compressed_indices) - i)).to_bytes(1, byteorder='little'))
            w.write(byte.to_bytes(1, byteorder='little'))

        w.write(bytes([0x00, 0x3b]))  # end of image data, end of GIF file


def decode(indices, color_table):
    """
    Decodes a matrix of indices to an image by mapping each index
    back to the corresponding element in the color table.
    """
    return color_table[indices]


def main(image_path, output_path, bitdepth, lut_method, dithering):
    """
    Do NOT change this function. Only complete the functions that raise
    NotImplementedError(). You can make additional functions if necessary.
    """
    assert (isinstance(bitdepth, int)), 'bitdepth needs to be an int.'
    assert (bitdepth <= 8 and bitdepth >= 1), 'bitdepth needs to be in range [1,8].'

    if dithering is None:
        dithering = 0

    if bitdepth is None:
        print("Warning: No bitdepth specified, using bitdepth 6.");
        bitdepth = 6

    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    # Make sure the image is always in int16 format and
    # has three dimensions: (h, w, 3)
    if img.dtype == bool:
        img = img.astype(np.int16) * 255
    else:
        img = img.astype(np.int16)
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, -1)
    print("Image dimensions:", img.shape)
    assert img.shape[2] == 3, 'program assumes three color channels.'

    color_table_size = 1 << bitdepth

    print("make color table...")
    color_table = None
    if lut_method == 'grayscale':
        color_table = make_grayscale_color_table(color_table_size)
    elif lut_method == 'random-colors':
        color_table = make_random_color_tabel(color_table_size)
    elif lut_method == 'random-sampling':
        color_table = make_random_sample_color_table(color_table_size, img)
    elif lut_method == 'median-cut':
        color_table = make_median_cut_color_table(color_table_size, img)
    elif lut_method == 'kmeans':
        color_table = make_kmeans_color_table(color_table_size, img)
    else:
        raise ValueError("Unknown LUT method " + str(lut_method))
    assert color_table_size == color_table.shape[0]

    print("transform...")
    indices = None
    if dithering == 0:
        indices = transform_image_to_indices_no_dithering(img, color_table)
    elif dithering == 1:
        indices = transform_image_to_indices_dithering_1(img, color_table)
    elif dithering == 2:
        indices = transform_image_to_indices_dithering_2(img, color_table)
    else:
        raise ValueError("Unknown dithering strategy: " + str(dithering))

    print("encode...")
    encode_gif(indices, color_table, output_path)

    print("decode...")
    img_coded = decode(indices, color_table)

    print("psnr...")
    psnr = calculate_psnr(img, img_coded)

    print('PSNR: %6.3f dB' % psnr)


if __name__ == '__main__':
    class CustomHelpFormatter(argparse.HelpFormatter):
        def _format_action_invocation(self, action):
            if not action.option_strings or action.nargs == 0:
                return super()._format_action_invocation(action)
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            return ', '.join(action.option_strings) + '   ' + args_string


    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    parser.add_argument('-i', '--image-path', type=str, required=True, help="input image (PNG)")
    parser.add_argument('-o', '--output-path', type=str, required=True, help="output file (GIF)")
    parser.add_argument('-b', '--bitdepth', type=int, required=True, help="bitdepth")
    parser.add_argument('-m', '--lut-method', type=str, required=False,
                        choices=['grayscale', 'random-colors', 'random-sampling', 'median-cut', 'kmeans'],
                        default='random-colors', help="LUT method")
    parser.add_argument('-d', '--dithering', type=int, required=False,
                        choices=[0, 1, 2], default=0,
                        help="dithering method (0 = no dithering, 1 = Floyd-Steinberg, 2 = MAE)")

    args = parser.parse_args()

    main(**vars(args))
