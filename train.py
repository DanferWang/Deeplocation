import json
import logging
from pathlib import Path
from datetime import datetime
from functools import partial
from typing import List
from shutil import copy

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet101V2

from dataloader import ImageDataset, Partitioning, get_partitionings


def build_multi_partitioning_model(partitionings: List[Partitioning], checkpoint: None) -> tf.keras.models.Model:
    """Build ResNet model with multiple classifier on top - one for each partitioning

    :param partitionings: list of partitionings
    :param checkpoint: path to model checkpoint

    """
    if not checkpoint:
        # train from scratch using imagenet weights
        base_model = ResNet101V2(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='avg')

        classifiers = []  # one classifier for each partitioning on top
        for p in partitionings:
            c = tf.keras.layers.Dense(p.nclasses, activation=None, name=p.shortname + '_logits')(base_model.output)
            c = tf.keras.activations.softmax(c)
            classifiers.append(c)

        # during optimization the losses for each classifier are added by default
        return tf.keras.models.Model(inputs=base_model.input, outputs=classifiers)

    else:
        base_model = tf.keras.models.load_model(checkpoint)
        base_model.summary()
        if all([base_model.get_layer(p.shortname + '_logits').get_weights()[1].shape[0] == p.nclasses for p in
                partitionings]):
            return base_model
        else:
            raise ValueError('Shape mismatch for given classification layers and size of partitioning')


def step_lr(epoch, base_lr, step_size, gamma=0.5):
    """Decay the learning rate every step_size epochs

    :param epoch: current epoch
    :param base_lr: learning rate on training start
    :param step_size: period of learning rate decay
    :param gamma: multiplicative factor of learning rate decay, default: 0.5
    :return: learning rate for current epoch
    """

    return base_lr * gamma ** (epoch // step_size)


def main(cfg):
    timestamp = datetime.now().strftime('%Y-%m-%d:%H-%M')
    result_dir = Path(cfg['base_result_dir'], timestamp)
    result_dir.mkdir(parents=True)

    # save partitionings to result folder
    for p_file in cfg['data']['partitioning']['filenames']:
        copy(Path(cfg['data']['partitioning']['base_folder'], p_file).resolve(), result_dir.resolve() / p_file)
    # copy config file
    with open(result_dir / 'cfg.json', 'w') as fw:
        json.dump(cfg, fw, indent=4)

    logging.info('Initialize the partitionings...')
    partitionings = get_partitionings(**cfg['data']['partitioning'])
    logging.info(partitionings)

    logging.info('Build the model...')
    tf.device('/gpu:0')
    # build model with n classifiers on top
    # the total loss that will be minimized by the model will be the sum of all individual losses
    model = build_multi_partitioning_model(
        partitionings=partitionings,
        checkpoint=cfg['training']['continue_from_checkpoint']
    )
    callbacks = []

    # learning rate scheduler
    if cfg['training']['lr_scheduler_activate']:
        schedule_func = partial(step_lr, base_lr=cfg['training']['lr'], **cfg['training']['lr_scheduler'])
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule_func)
        callbacks.append(lr_scheduler)

    # optimizer
    optim = tf.keras.optimizers.SGD(learning_rate=cfg['training']['lr'], momentum=0.9, decay=1e-4, nesterov=True)
    model.compile(optimizer=optim, loss=tf.keras.losses.sparse_categorical_crossentropy)

    logging.info('Read the training set...')
    dataset_train = ImageDataset(**cfg['data']['train'], partitionings=partitionings)
    logging.info(f'Number of images for training: {len(dataset_train)}')
    logging.info(f'Number of batches for training: {dataset_train.nbatches}')

    logging.info('Read the validation set...')
    dataset_valid = ImageDataset(**cfg['data']['valid'], validation=True, partitionings=partitionings)

    # tensorboard logging
    log_dir = result_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    tbc = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), update_freq=100, write_graph=False, profile_batch=0)
    callbacks.append(tbc)

    # save best model callback
    if cfg['training']['save_checkpoints']:
        cc = tf.keras.callbacks.ModelCheckpoint(
            str(result_dir / 'model-{epoch:02d}-{val_loss:.2f}.h5'),
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=False,
            save_freq='epoch'
        )
        callbacks.append(cc)

    # train the model
    model.fit(
        dataset_train.ds,
        callbacks=callbacks,
        steps_per_epoch=dataset_train.nbatches,
        validation_data=dataset_valid.ds,
        **cfg['training']['fit_params']
    )

    return

if __name__ == '__main__':
    # restrict GPU Mem
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 9GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9216)])
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    # user define
    root_path = "./Deeplocation/"

    config_file = root_path + "example.json"
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d-%m-%Y %H:%M:%S',
                        level=logging.INFO)

    cfg = json.load(open(config_file, 'r'))

    main(cfg)
