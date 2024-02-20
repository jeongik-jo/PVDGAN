import HyperParameters as hp
import Train
import time
import Models
import Dataset
import Evaluate
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt


def train_gan():
    enc = Models.Encoder()
    dec = Models.Decoder()
    dis = Models.Discriminator()

    if hp.load_model:
        enc.load(), dec.load(), dis.load()

    train_dataset = Dataset.load_train_dataset()
    test_dataset = Dataset.load_test_dataset()

    results = {}
    for epoch in range(hp.epochs):
        print(datetime.datetime.now())
        print('epoch', epoch)
        start = time.time()

        train_results = Train.train(enc.model, dec.model, dis.model, train_dataset)
        print('saving...')
        enc.to_ema()
        dec.to_ema()
        enc.save()
        dec.save()
        dis.save()
        dec.save_imgs(enc.model, Dataset.load_sample_dataset(), epoch)
        print('saved')
        print('time: ', time.time() - start, '\n')

        if hp.eval_model and (epoch + 1) % hp.epoch_per_eval == 0:
            print('evaluating...')
            start = time.time()
            eval_results = Evaluate.eval(enc.model, dec.model, test_dataset)
            for key in train_results:
                try:
                    results[key].append(train_results[key])
                except KeyError:
                    results[key] = [train_results[key]]
            for key in eval_results:
                try:
                    results[key].append(eval_results[key])
                except KeyError:
                    results[key] = [eval_results[key]]

            print('evaluated')
            print('time: ', time.time() - start, '\n')
            if not os.path.exists('results/figures'):
                os.makedirs('results/figures')
            for key in results:
                np.savetxt('results/figures/%s.txt' % key, results[key], fmt='%f')
                plt.title(key)
                plt.xlabel('Epochs')
                plt.ylabel(key)
                plt.plot([(i + 1) * hp.epoch_per_eval for i in range(len(results[key]))], results[key])
                plt.savefig('results/figures/%s.png' % key)
                plt.clf()

        enc.to_train()
        dec.to_train()


train_gan()

