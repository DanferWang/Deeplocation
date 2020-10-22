import argparse
import os
from pathlib import Path
import pandas as pd
import sys
import csv
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses unnecessarily excessive console output
import tensorflow as tf

# own imports
import utils
import geo_estimation


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    # dataset
    parser.add_argument('-l', '--labels', type=str, default='/dbfs/mnt/group03/dataset_0e_merge_drope.csv/part-00000-tid-2991617681581727343-288f20d1-597c-4be2-a8cd-f040575e7279-189207-1-c000.csv', help='path to ground truth labels')
    #parser.add_argument('-i', '--inputs', nargs='+', type=str, required=True, help='path to image file(s)')
    # model
    parser.add_argument('-m', '--model', type=Path, default='/dbfs/mnt/group03/inference_dataset0/model_ten-03-20.87.h5', help='path to a model checkpoint (.h5)')
    # output_dir
    parser.add_argument('-o', '--output', type=str, default= "/dbfs/mnt/group03", help="path to output directory")
    args = parser.parse_args()
    return args

def writeoutput(args, num_predict, res_list, out_p):
    if not os.path.exists(out_p):
        os.makedirs(out_p)
    # output csv file name
    fname = f"dataset0_inference_result.csv"

    with open(os.path.join(out_p, fname), 'w') as f:
        res_writer = csv.writer(f, delimiter=',')
        res_writer.writerow(['img_id', 'gt_lat', 'predicted_lat', 'gt_long', 'predicted_long','great_circle_distance','url'])

        for row in res_list:
            res_writer.writerow(row)


def main():
    # load arguments
    args = parse_args()

    # check if gpu is available
    if len(tf.config.list_physical_devices('GPU')) == 0:
        print('No GPU available. Using CPU instead ... ')
        args.cpu = True

    # restrict GPU Mem
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 9GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    ge_base = geo_estimation.GeoEstimator(args.model, scope=args.model.parent.name, use_cpu=args.cpu)

    if args.labels:  # read labels (if specified)
        meta_info = pd.read_csv(args.labels)
    else:  # create empty dataframe
        meta_info = pd.DataFrame(columns=['IMG_ID', 'LAT', 'LON'])

    predict_images = pd.read_csv(args.labels, usecols=["IMG_ID"])
    image_url = pd.read_csv(args.labels, usecols=["s3_http"])
    #predict_images = predict_images.iloc[0:]
    print(predict_images.iloc[0].values[0])
    # get predictions

    gc_dists = {}
    resultlist =[]
    i = 0
    for j in range(len(predict_images)):
        i += 1
        print('{} / {} Processing: {}'.format(i, len(predict_images), predict_images.iloc[j].values[0]))

        # get meta information if available
        # img_dir
        img_path = "/dbfs/mnt/multimedia-commons/data/images/" + predict_images.iloc[j].values[0]
        fname = os.path.basename("/dbfs/mnt/multimedia-commons/data/images/" + predict_images.iloc[j].values[0])
        img_meta = meta_info.loc[meta_info['IMG_ID'] == fname]
        if len(img_meta) > 0:
            img_meta = img_meta.iloc[0]
        else:
            img_meta = {}

        ge = ge_base

        print('\t--> Using {} network for geolocation'.format(ge.network_dict['scope']))
        ge.calc_output_dict(img_path)

        print('\t### GEOESTIMATION RESULTS ###')
        # ======== modify here 10/15 ========
        # ===== len -1 =====
        for p in range(len(ge.network_dict['partitionings'])):
            p_name = ge.network_dict['partitionings'][p]
            pred_loc = ge.output_dict['predicted_GPS_coords'][p]

            # only calculate result if ground truth location is specified in args.labels
            dist_str = ''
            if 'LAT' in img_meta and 'LON' in img_meta:
                if p_name not in gc_dists:
                    gc_dists[p_name] = {}
                gc_dists[p_name][fname] = utils.gc_distance(pred_loc, [img_meta['LAT'], img_meta['LON']])
                dist_str = f' --> GCD to true location: {gc_dists[p_name][fname]:.2f} km'

            print(f"\tPredicted GPS coordinate (lat, lng) for <{p_name}>: ({pred_loc[0]:.2f}, {pred_loc[1]:.2f})" +
                  dist_str)
            if p == 3:
                resultlist.append([predict_images.iloc[j].values[0], format(img_meta['LAT'], '.2f') , format(pred_loc[0],'.2f'), format(img_meta['LON'],'.2f'), format(pred_loc[1],'.2f'), format(gc_dists[p_name][fname],'.2f'), image_url.iloc[j].values[0]])

    # print results for all files with specified gt location
    if args.labels:
        print('### TESTSET RESULTS ###')
        utils.print_results(gc_dists)

    writeoutput(args, len(predict_images), resultlist, args.output)


if __name__ == '__main__':
    sys.exit(main())
