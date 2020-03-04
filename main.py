import numpy as np
import torch
import torch.nn as nn1
import matplotlib.pyplot as plt

from utils import *
from model import *
from mh import *
# from train import *

def jsd_experiment(gen, disc, sample_size=10000, num_tests=5, mh=True, std=0.05, mu=[-2,-1,0,1,2]):
    jsd_vals = []

    for i in range(num_tests):
        if mh:
            with torch.no_grad():
                initials = gaussian_mix_generator(sample_size).to(0, torch.float)
                samples = generate_vectorized_mh_samples(sample_size, gen, disc, 640, initials, batch_size=sample_size).cpu().detach().numpy()
                jsd_vals.append(jsd(samples))
        else:
            samples = gen(generate_noise(sample_size, gen.module.nz)).cpu().detach().numpy()
            jsd_vals.append(jsd(samples))
    jsd_vals = np.array(jsd_vals)
    print(jsd_vals)
    return jsd_vals.mean(), jsd_vals.std()

nz=2

gen = nn.DataParallel(GaussianGenerator(nz, 100).cuda(), device_ids=(0,1))
disc = nn.DataParallel(GaussianDiscriminator(100).cuda(), device_ids=(0,1))
gen.load_state_dict(torch.load('models/badgen'))
disc.load_state_dict(torch.load('models/baddisc'))

# plot_gaussian_mix(1000)
base_samples = gen(generate_noise(10000, nz)).cpu().detach().numpy()
print(gaussian_metric(base_samples))
print(jsd(base_samples))
plot_2d(base_samples, boundaries=False, title='Without Metropolis Hastings')
# plot_2d(base_samples, boundaries=True)


# print(test_calibration_gaussian(gen, disc, 10000))


#
num = 20000
train_data = gaussian_mix_generator(num)
points = gen(generate_noise(num, nz))

data = torch.cat((train_data.to(0, torch.float), points.to(0, torch.float)), dim=0)
labels = torch.cat((torch.full((num, 1), 1).cuda(), torch.full((num, 1), 0).cuda()), dim=0)
predictions = disc(data)
predictions = predictions.cpu().detach().numpy().reshape(-1)
labels = labels.cpu().detach().numpy().reshape(-1).astype(bool)
calibrator = Isotonic()
calibrator.fit(predictions, labels)
del data
del predictions
del labels
cal_d = lambda input: torch.from_numpy(calibrator.predict(disc(input).cpu().detach().numpy().reshape(-1))).cuda()



# difference between calibrated and uncalibrated

sample_size = 10000

# mean, std = jsd_experiment(gen, cal_d, sample_size=sample_size)


#
# initials = gaussian_mix_generator(sample_size).to(0, torch.float)
# samples = generate_mh_samples(sample_size, gen, disc, 400, initials).cpu().detach().numpy()
# print(gaussian_metric(samples))
# print(jsd(samples))
# --------------- vectorize below here
# initials = gaussian_mix_generator(sample_size).to(0, torch.float)
# samples = generate_vectorized_mh_samples(sample_size, gen, disc, 400, initials, batch_size=1000).cpu().detach().numpy()
# print(samples)
# print(gaussian_metric(samples))
# print(jsd(samples))
# plot_2d(samples, boundaries=False)


# initials = gaussian_mix_generator(sample_size).to(0, torch.float)
# cal_samples = generate_mh_samples(sample_size, gen, cal_d, 400, initials).cpu().detach().numpy()
# print(gaussian_metric(cal_samples))
# print(jsd(cal_samples))
# plot_2d(cal_samples, boundaries=False)
# plot_2d(cal_samples, boundaries=True)
# --------------- vectorize below here
with torch.no_grad():
    initials = gaussian_mix_generator(sample_size).to(0, torch.float)
    cal_samples = generate_vectorized_mh_samples(sample_size, gen, cal_d, 640, initials, batch_size=sample_size).cpu().detach().numpy()
    print(cal_samples)
    print(len(cal_samples))
    print(gaussian_metric(cal_samples))
    print(jsd(cal_samples))
    plot_2d(cal_samples, boundaries=False, title='With Metropolis-Hastings')


#
# print(jsd_experiment(gen, cal_d, sample_size, mh=False))
# print(jsd_experiment(gen, cal_d, sample_size))

# with torch.no_grad():
#     jsd_store = []
#     high_quality_store = []
#     for K in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]:
#         initials = gaussian_mix_generator(sample_size).to(0, torch.float)
#         cal_samples = generate_vectorized_mh_samples(sample_size, gen, cal_d, K, initials, batch_size=sample_size).cpu().detach().numpy()
#         # print(gaussian_metric(cal_samples))
#         high_quality_store.append(gaussian_metric(cal_samples)[1])
#         jsd_store.append(jsd(cal_samples))

# fig, axes = plt.subplots(1, 2)
# axes[0].plot([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200], high_quality_store)
# axes[1].plot([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200], jsd_store)
# print(high_quality_store)
# print(jsd_store)
# plt.show()



# fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
# plot_2d(base_samples, axes[0])
# axes[0].set_title("No MH")
# plot_2d(samples, axes[1])
# axes[1].set_title('MH - no calibration')
# plot_2d(cal_samples, axes[2])
# axes[2].set_title('MH - isotonic calibration')
# plt.show()


#
#
# plot_calibration_curve(gen, cal_d, 10000, numpy=True)
# plot_calibration_curve(gen, disc, 10000, numpy=False)
# print(test_calibration_gaussian(gen, cal_d, 1000, numpy=True))
