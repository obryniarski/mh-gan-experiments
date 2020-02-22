import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import *
from utils import *

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


def train_gan(gen, disc, data, epochs, bs, lr, display=False):

    gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

    disc_opt = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    loss = nn.BCELoss()
    fake_label = 0
    real_label = 1
    nz = gen.module.nz
    if display:
        points_super = []
        display_epochs = list(np.linspace(0, epochs, 6, dtype=int, endpoint=False))
        print(display_epochs)


    data_gen = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=True)

    for epoch in range(epochs):
        print('----- Epoch %i -----' % epoch)

        for batch in data_gen:
            batch.to(0)

            # discriminator training
            disc.zero_grad()
            reals = batch
            labels = torch.full((batch.shape[0], 1), real_label, device=0)
            l1 = loss(disc(reals), labels)

            noise = generate_noise(batch.shape[0], nz)
            fakes = gen(noise)
            labels = torch.full((batch.shape[0], 1), fake_label, device=0)
            l2 = loss(disc(fakes), labels)
            d_total_l = l1 + l2
            d_total_l.backward()
            disc_opt.step()

            # generator training
            gen.zero_grad()
            noise = generate_noise(batch.shape[0], nz)
            fakes = gen(noise)
            labels = torch.full((batch.shape[0], 1), real_label, device=0)
            g_total_l = loss(disc(fakes), labels)
            g_total_l.backward()
            gen_opt.step()

        if display and epoch in display_epochs:
            points_super.append(gen(generate_noise(10000, nz)).cpu().detach().numpy())


        print('disc loss: %.3f' % (d_total_l))
        print('gen loss: %.3f' % (g_total_l))

    if display:
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 8))
        for i in range(len(points_super)):
            axes[i // 3, i % 3].scatter(points_super[i][:, 0], points_super[i][:, 1], s=1.5, alpha=0.3)
            axes[i // 3, i % 3].set_title('Epoch %i - %.4f' % (display_epochs[i], jsd(points_super[i])))

        plt.show()



    return gen, disc


gaussian_data = gaussian_mix_generator(64000).to(0, torch.float)

base_gen = nn.DataParallel(GaussianGenerator(2, 100).cuda(), device_ids=(0,1))
base_disc = nn.DataParallel(GaussianDiscriminator(100).cuda(), device_ids=(0,1))

trained_gan, trained_disc = train_gan(base_gen, base_disc, gaussian_data, epochs=25, bs=256, lr=0.0005, display=True)

torch.save(trained_gan.state_dict(), 'models/badgen')
torch.save(trained_disc.state_dict(), 'models/baddisc')
