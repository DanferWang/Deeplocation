import csv
import os
import logging
import sys
import pandas as pd
import s2sphere as s2
from time import time
from functools import partial
from multiprocessing import Pool
from collections import Counter


def _init_parallel(img, level):
    cell = create_s2_cell(img[1], img[0])
    hexid = create_cell_at_level(cell, level)
    return [*img, hexid, cell]


def init_cells(img_container_0, level):
    start = time()
    f = partial(_init_parallel, level=level)
    img_container = []
    with Pool(8) as p:
        for x in p.imap_unordered(f, img_container_0, chunksize=1000):
            img_container.append(x)
    logging.debug(f'Time multiprocessing: {time() - start:.2f}s')
    start = time()
    h = dict(Counter(list(list(zip(*img_container))[3])))
    logging.debug(f'Time creating h: {time() - start:.2f}s')

    return img_container, h


def delete_cells(img_container, h, t_min):
    del_cells = {k for k, v in h.items() if v <= t_min}
    h = {k: v for k, v in h.items() if v > t_min}
    img_container_f = []
    for img in img_container:
        hexid = img[3]
        if hexid not in del_cells:
            img_container_f.append(img)
    return img_container_f, h


def gen_subcells(img_container_0, h_0, level, t_max):
    img_container = []
    h = {}
    for img in img_container_0:
        hexid_0 = img[3]
        if h_0[hexid_0] > t_max:
            hexid = create_cell_at_level(img[4], level)
            img[3] = hexid
            try:
                h[hexid] = h[hexid] + 1
            except:
                h[hexid] = 1
        else:
            try:
                h[hexid_0] = h[hexid_0] + 1
            except:
                h[hexid_0] = 1
        img_container.append(img)
    return img_container, h


def create_s2_cell(lat, lng):
    p1 = s2.LatLng.from_degrees(lat, lng)
    cell = s2.Cell.from_lat_lng(p1)
    return cell


def create_cell_at_level(cell, level):
    cell_parent = cell.id().parent(level)
    hexid = cell_parent.to_token()
    return hexid


def write_output(cell_img_min, cell_img_max, img_container, h, num_images, out_p):
    fname = f"cells_{cell_img_min}_{cell_img_max}_images_{num_images}.csv"
    logging.info(f'Write to {os.path.join(out_p, fname)}')
    with open(os.path.join(out_p, fname), 'w') as f:
        cells_writer = csv.writer(f, delimiter=',')
        # write column names
        cells_writer.writerow(['class_label', 'hex_id', 'imgs_per_cell', 'latitude_mean', 'longitude_mean'])

        # write dict
        i = 0
        cell2class = {}
        coords_sum = {}

        # generate class ids for each hex cell id
        for k in h.keys():
            cell2class[k] = i
            coords_sum[k] = [0, 0]
            i = i + 1

        # calculate mean GPS coordinate in each cell
        for img in img_container:
            coords_sum[img[3]][0] = coords_sum[img[3]][0] + img[1]
            coords_sum[img[3]][1] = coords_sum[img[3]][1] + img[0]

        # write partitioning information
        for k, v in h.items():
            cells_writer.writerow([cell2class[k], k, v, coords_sum[k][0] / v, coords_sum[k][1] / v])


def main():
    # user define
    input_dataset = "/dbfs/mnt/group03/face_less0_ten.csv"
    output_path = "/dbfs/mnt/group03/face_less0_ten_create_cell"
    cell_level_min = 2
    cell_level_max = 30
    cell_img_min = 5
    cell_img_max = 500 # 100, 200, 500

    level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=level)

    # read dataset
    df = pd.read_csv(input_dataset, usecols=['LON', 'LAT', 'IMG_ID'])
    img_container = list(df.itertuples(index=False, name=None))
    num_images = len(img_container)
    logging.info('{} images available.'.format(num_images))
    level = cell_level_min

    # create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize
    logging.info('Initialize cells of level {} ...'.format(level))
    start = time()
    img_container, h = init_cells(img_container, level)
    logging.info(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')

    logging.info('Remove cells with |img| < t_min ...')
    start = time()
    img_container, h = delete_cells(img_container, h, cell_img_min)
    logging.info(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')

    logging.info('Create subcells ...')
    while any(v > cell_img_max for v in h.values()) and level < cell_level_max:
        level = level + 1
        logging.info('Level {}'.format(level))
        start = time()
        img_container, h = gen_subcells(img_container, h, level, cell_img_max)
        logging.info(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')

    logging.info('Remove cells with |img| < t_min ...')
    start = time()
    img_container, h = delete_cells(img_container, h, cell_img_min)
    logging.info(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')
    logging.info(f'Number of images: {len(img_container)}')

    logging.info('Write output file ...')
    write_output(cell_img_min, cell_img_max, img_container, h, num_images, output_path)


if __name__ == '__main__':
    sys.exit(main())
