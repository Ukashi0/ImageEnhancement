import torch
import numpy as np
import os
import time
from opts.train_opt import TrainOptions
from util.dataLoader import CreateDataLoader
from models import model

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

if __name__ == '__main__':
    opt = TrainOptions().parse()
    config = get_config(opt.config)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(dataset)
    model = model.GanModel()
    model.initialize(opt)
    total_steps = 0
    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            model.set_input(data)
            model.optimize_parameters(epoch)


            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors(epoch)
                t = (time.time() - iter_start_time) / opt.batchSize

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if opt.new_lr:
            if epoch == opt.niter:
                model.update_learning_rate()
            elif epoch == (opt.niter + 20):
                model.update_learning_rate()
            elif epoch == (opt.niter + 70):
                model.update_learning_rate()
            elif epoch == (opt.niter + 90):
                model.update_learning_rate() #
                model.update_learning_rate()
                model.update_learning_rate()
                model.update_learning_rate()
        else:
            if epoch > opt.niter:
                model.update_learning_rate()




