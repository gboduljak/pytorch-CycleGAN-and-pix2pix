"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
from pathlib import Path

import torch
import torch_fidelity

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from translate import (create_train_dataset, create_val_dataset, save_images,
                       select_visuals)
from util.visualizer import Visualizer

# def seed_everything(seed: int):
#   import os
#   import random

#   import numpy as np
#   import torch

#   random.seed(seed)
#   os.environ['PYTHONHASHSEED'] = str(seed)
#   np.random.seed(seed)
#   torch.manual_seed(seed)
#   torch.cuda.manual_seed(seed)
#   # torch.mps.manual_seed(seed)


if __name__ == '__main__':
  # seed_everything(42)
  # print("seeded")

  opt = TrainOptions().parse()   # get training options
  # create a dataset given opt.dataset_mode and other options
  train_dataset = create_dataset(opt)
  train_dataset_without_augmentations = create_train_dataset(opt)
  val_dataset = create_val_dataset(opt)
  # get the number of images in the dataset.
  dataset_size = len(train_dataset)
  print('The number of training images = %d' % dataset_size)

  # create a model given opt.model and other options
  model = create_model(opt)
  # regular setup: load and print networks; create schedulers
  model.setup(opt)
  # create a visualizer that display/save images and plots
  visualizer = Visualizer(opt)
  total_iters = 0                # the total number of training iterations
  smallest_val_fid = float('inf')
  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
  for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    # the number of training iterations in current epoch, reset to 0 every epoch
    epoch_iter = 0
    # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    visualizer.reset()
    for i, data in enumerate(train_dataset):  # inner loop within one epoch
      iter_start_time = time.time()  # timer for computation per iteration
      if total_iters % opt.print_freq == 0:
        t_data = iter_start_time - iter_data_time

      total_iters += opt.batch_size
      epoch_iter += opt.batch_size
      # unpack data from dataset and apply preprocessing
      model.set_input(data)
      # calculate loss functions, get gradients, update network weights
      model.optimize_parameters()

      if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
        save_result = total_iters % opt.update_html_freq == 0
        model.compute_visuals()
        visualizer.display_current_results(
            model.get_current_visuals(), epoch, save_result)

      if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
        losses = model.get_current_losses()
        t_comp = (time.time() - iter_start_time) / opt.batch_size
        visualizer.print_current_losses(
            epoch, epoch_iter, losses, t_comp, t_data)
        if opt.display_id > 0:
          visualizer.plot_current_losses(
              epoch, float(epoch_iter) / dataset_size, losses)

      if total_iters % opt.val_freq == 0:
        model.eval()

        model_translations_dir = Path(
            opt.checkpoints_dir, opt.name, 'translations')

        if not os.path.exists(model_translations_dir):
          os.mkdir(model_translations_dir)

        model_train_translations_dir = Path(model_translations_dir, 'train')
        model_val_translations_dir = Path(model_translations_dir, 'val')

        if not os.path.exists(model_train_translations_dir):
          os.mkdir(model_train_translations_dir)
        if not os.path.exists(model_val_translations_dir):
          os.mkdir(model_val_translations_dir)

        model_with_iter_train_translations_dir = Path(
            model_train_translations_dir,
            'iter_%07d' % total_iters
        )
        model_with_iter_val_translations_dir = Path(
            model_val_translations_dir,
            'iter_%07d' % total_iters
        )
        if not os.path.exists(model_with_iter_train_translations_dir):
          os.mkdir(model_with_iter_train_translations_dir)
        if not os.path.exists(model_with_iter_val_translations_dir):
          os.mkdir(model_with_iter_val_translations_dir)

        print('translating train...')
        with torch.no_grad():
          for i, data in enumerate(train_dataset_without_augmentations):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = select_visuals(
                model.get_current_visuals(),
                opt.direction
            )  # get image results
            img_path = model.get_image_paths()     # get image paths
            save_images(
                image_dir=model_with_iter_train_translations_dir,
                visuals=visuals,
                image_path=img_path
            )
            if i % 10 == 0:
              print('processing (%04d)-th image... %s' % (i, img_path))

        print('translating val...')
        with torch.no_grad():
          for i, data in enumerate(val_dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = select_visuals(
                model.get_current_visuals(),
                opt.direction
            )  # get image results
            img_path = model.get_image_paths()     # get image paths
            save_images(
                image_dir=model_with_iter_val_translations_dir,
                visuals=visuals,
                image_path=img_path
            )
            if i % 10 == 0:
              print('processing (%04d)-th image... %s' % (i, img_path))

        target_real_train_dir = os.path.join(
            opt.dataroot,
            'trainB'
        )
        target_real_val_dir = os.path.join(
            opt.dataroot,
            'valB'
        )

        train_metrics = torch_fidelity.calculate_metrics(
            input1=str(target_real_train_dir),
            # fake dir,
            input2=str(Path(model_with_iter_train_translations_dir)),
            fid=True,
            verbose=False,
            cuda=torch.cuda.is_available(),
        )

        val_metrics = torch_fidelity.calculate_metrics(
            input1=str(target_real_val_dir),
            input2=str(Path(model_with_iter_val_translations_dir)),  # fake dir,
            fid=True,
            verbose=False,
            cuda=torch.cuda.is_available(),
        )

        train_log_file = os.path.join(
            Path(opt.checkpoints_dir, opt.name), 'train_log.txt')
        val_log_file = os.path.join(
            Path(opt.checkpoints_dir, opt.name), 'val_log.txt')
        smallest_val_fid_file = os.path.join(
            Path(opt.checkpoints_dir, opt.name), 'smallest_val_fid.txt')

        with open(train_log_file, 'a') as tl:
          tl.write(f'iter: {total_iters}\n')
          tl.write(
              f'frechet_inception_distance: {train_metrics["frechet_inception_distance"]}\n'
          )

        with open(val_log_file, 'a') as tl:
          tl.write(f'iter: {total_iters}\n')
          tl.write(
              f'frechet_inception_distance: {val_metrics["frechet_inception_distance"]}\n'
          )

        if val_metrics['frechet_inception_distance'] < smallest_val_fid:
          smallest_val_fid = val_metrics['frechet_inception_distance']
          print('saving the smallest_val_fid model')
          model.save_networks('smallest_val_fid')

          if os.path.exists(smallest_val_fid_file):
            os.remove(smallest_val_fid_file)

          with open(smallest_val_fid_file, 'a') as tl:
            tl.write(
                f'iter: {total_iters}\n'
            )
            tl.write(
                f'frechet_inception_distance: {val_metrics["frechet_inception_distance"]}\n'
            )

        model.train()

      if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
        print('saving the latest model (epoch %d, total_iters %d)' %
              (epoch, total_iters))
        # it's useful to occasionally show the experiment name on console
        print(opt.name)
        model.save_networks('latest')
        model.save_networks('iter_%07d' % total_iters)

      iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
      print('saving the model at the end of epoch %d, iters %d' %
            (epoch, total_iters))
      model.save_networks('latest')
      model.save_networks(epoch)

    # update learning rates in the beginning of every epoch, except the first epoch.
    model.update_learning_rate()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
