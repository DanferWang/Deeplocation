import logging
import pandas as pd
import s2sphere as s2
from pathlib import Path
from typing import Union
from tqdm import tqdm

tqdm.pandas()


def create_s2_cell(latlng):
    p1 = s2.LatLng.from_degrees(latlng['lat'], latlng['lng'])
    cell = s2.Cell.from_lat_lng(p1)
    return cell


def get_id_s2cell_mapping_from_raw(csv_file, col_img_id, col_lat, col_lng) -> pd.DataFrame:
    usecols = [col_img_id, col_lat, col_lng]
    df = pd.read_csv(csv_file, usecols=usecols)
    df = df.rename(columns={k: v for k, v in zip(usecols, ['img_path', 'lat', 'lng'])})

    logging.info('Initialize s2 cells...')
    df['s2cell'] = df[['lat', 'lng']].progress_apply(create_s2_cell, axis=1)
    df = df.set_index(df['img_path'])
    return df[['s2cell']]


def assign_class_index(cell: s2.Cell, mapping: dict) -> Union[int, None]:
    for l in range(2, 30):
        cell_parent = cell.id().parent(l)
        hexid = cell_parent.to_token()
        if hexid in mapping:
            return int(mapping[hexid])  # class index

    return None  # valid return since not all regions are covered


if __name__ == '__main__':
    # user define
    cell_path = "dbfs/mnt/group03/face_less0_ten_create_cell/"
    input_dataset = "/dbfs/mnt/group03/face_less0_ten.csv"
    output_path = "dbfs/mnt/group03/face_less0_ten_assign_classes.csv"
    partionings_file_list = [cell_path + "cells_5_100_images_148001.csv",
                             cell_path + "cells_5_200_images_148001.csv",
                             cell_path + "cells_5_500_images_148001.csv"]

    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S',
        level=logging.INFO)

    output_file = Path(output_path)
    if output_file.exists():
        raise FileExistsError('Already created')
    output_file.parent.mkdir(exist_ok=True, parents=True)

    logging.info('Load CSV and initialize s2 cells')
    logging.info(f'Column image path: {"IMG_ID"}')
    logging.info(f'Column latitude: {"LAT"}')
    logging.info(f'Column longitude: {"LON"}')
    df_mapping = get_id_s2cell_mapping_from_raw(
        input_dataset,
        col_img_id="IMG_ID",
        col_lat="LAT",
        col_lng="LON"
    )

    partitioning_files = [Path(p) for p in partionings_file_list]
    for partitioning_file in partitioning_files:
        column_name = partitioning_file.name.split('.')[0]
        logging.info(f'Processing partitioning: {column_name}')
        partitioning = pd.read_csv(partitioning_file, encoding='utf-8', index_col='hex_id', skiprows=None)

        # create column with class indexes for respective partitioning
        mapping = partitioning['class_label'].to_dict()
        df_mapping[column_name] = df_mapping['s2cell'].progress_apply(lambda cell: assign_class_index(cell, mapping))
        nans = df_mapping[column_name].isna().sum()
        logging.info(f'Cannot assign a hexid for {nans} of {len(df_mapping.index)} images '
                     f'({nans / len(df_mapping.index) * 100:.2f}%)')

    # drop unimportant information
    df_mapping = df_mapping.drop(columns=['s2cell'])
    logging.info('Remove all images that could not be assigned a cell')
    original_dataset_size = len(df_mapping.index)
    df_mapping = df_mapping.dropna()

    for partitioning_file in partitioning_files:
        column_name = partitioning_file.name.split('.')[0]
        df_mapping[column_name] = df_mapping[column_name].astype('int32')

    fraction = len(df_mapping.index) / original_dataset_size * 100
    logging.info(f'Final dataset size: {len(df_mapping.index)}/{original_dataset_size} ({fraction:.2f})% from original')

    logging.info('Sort by image path...')
    df_mapping = df_mapping.sort_values(by='img_path')
    # store final dataset to file
    logging.info(f'Store dataset to {output_file}')
    df_mapping.to_csv(output_file, encoding='utf-8', index_label='img_path')

    exit(0)
