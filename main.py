import numpy as np
import torch
import torch.nn as nn1
import matplotlib.pyplot as plt

from utils import *
from model import *
from mh import *
# from train import *

nz=2

gen = nn.DataParallel(GaussianGenerator(nz, 100).cuda(), device_ids=(0,1))
disc = nn.DataParallel(GaussianDiscriminator(100).cuda(), device_ids=(0,1))
gen.load_state_dict(torch.load('models/badgen'))
disc.load_state_dict(torch.load('models/baddisc'))

# plot_gaussian_mix(1000)
base_samples = gen(generate_noise(250, nz)).cpu().detach().numpy()
plot_2d(base_samples, boundaries=False)
plot_2d(base_samples, boundaries=True)
print(gaussian_metric(base_samples))
print(jsd(base_samples))

# print(test_calibration_gaussian(gen, disc, 10000))


#
num = 100000
train_data = gaussian_mix_generator(num)
points = gen(generate_noise(num, nz))

data = torch.cat((train_data.to(0, torch.float), points.to(0, torch.float)), dim=0)
labels = torch.cat((torch.full((num, 1), 1).cuda(), torch.full((num, 1), 0).cuda()), dim=0)
predictions = disc(data)
predictions = predictions.cpu().detach().numpy().reshape(-1)
labels = labels.cpu().detach().numpy().reshape(-1).astype(bool)
calibrator = Isotonic()
calibrator.fit(predictions, labels)
cal_d = lambda input: calibrator.predict(disc(input).cpu().detach().numpy().reshape(-1))



# difference between calibrated and uncalibrated

sample_size = 250
#
# initials = gaussian_mix_generator(sample_size).to(0, torch.float)
# samples = generate_mh_samples(sample_size, gen, disc, 400, initials).cpu().detach().numpy()
# print(gaussian_metric(samples))
# print(jsd(samples))

initials = gaussian_mix_generator(sample_size).to(0, torch.float)
cal_samples = generate_mh_samples(sample_size, gen, cal_d, 400, initials).cpu().detach().numpy()
plot_2d(cal_samples, boundaries=False)
plot_2d(cal_samples, boundaries=True)
print(gaussian_metric(cal_samples))
print(jsd(cal_samples))


# fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
# plot_2d(base_samples, axes[0])
# axes[0].set_title("No MH")
# plot_2d(samples, axes[1])
# axes[1].set_title('MH - no calibration')
# plot_2d(cal_samples, axes[2])
# axes[2].set_title('MH - isotonic calibration')
# plt.show()


# different K values
# sample_size = 100
# for K in range(100, 601, 50):
#     initials = gaussian_mix_generator(sample_size).to(0, torch.float)
#     cal_samples = generate_mh_samples(sample_size, gen, cal_d, K, initials).cpu().detach().numpy()
#     print(gaussian_metric(cal_samples))
#     print(jsd(cal_samples))


#
#
# plot_calibration_curve(gen, cal_d, 10000, numpy=True)
# plot_calibration_curve(gen, disc, 10000, numpy=False)
# print(test_calibration_gaussian(gen, cal_d, 1000, numpy=True))
