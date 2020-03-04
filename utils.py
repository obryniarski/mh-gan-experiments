import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def gaussian_mix_generator(num_generated=64000, std=0.05, mu=[-2, -1, 0, 1, 2]):

    # assert num_generated % len(mu) ** 2 == 0
    num_each = num_generated // len(mu) ** 2
    data = []
    # for x in mu:
    #     for y in mu:
    #         data.append(np.random.multivariate_normal([x, y], [[std ** 2, 0], [0, std ** 2]], size=num_each))
    for _ in range(num_generated):
        x = np.random.randint(5)
        y = np.random.randint(5)
        sample = torch.from_numpy(np.random.multivariate_normal([mu[x], mu[y]], [[std**2, 0],[0, std**2]], size=(1)))
        data.append(sample)


    return torch.cat(data, dim=0)


def plot_gaussian_mix(num_generated=64000, std=0.05, mu=[-2,-1,0,1,2]):
    data = gaussian_mix_generator(num_generated, std, mu)
    plt.scatter(data[:, 0], data[:, 1], s=1.5, alpha=0.3)
    plt.show()

def plot_2d(points, axis=None, boundaries=False, std=0.05, mu=[-2,-1,0,1,2]):
    if axis:
        axis.grid()
        axis.scatter(points[:, 0], points[:, 1], s=1.5, alpha=0.3)
        if boundaries:
            for x in mu:
                for y in mu:
                    axis.add_artist(plt.Circle((x, y), 4 * std, color='r', fill=False, linewidth=0.15))

    else:
        fig, ax = plt.subplots()
        ax.grid()
        ax.scatter(points[:, 0], points[:, 1], s=1.5, alpha=0.3)
        if boundaries:
            for x in mu:
                for y in mu:
                    ax.add_artist(plt.Circle((x, y), 4 * std, color='r', fill=False, linewidth=0.15))
        plt.show()

def generate_noise(num_generated, nz):
    # return torch.clamp(torch.randn((num_generated, nz), dtype=torch.float, device=0), min=0)
    # return torch.randn((num_generated, nz), dtype=torch.float, device=0)
    return torch.rand((num_generated, nz), dtype=torch.float, device=0) * 2 - 1


def z_calibration_stat(disc, data, y, numpy):
    if numpy:
        return np.sum(y - disc(data)) / (np.sum(disc(data) *(1-disc(data)))) ** 0.5
    return torch.sum(y - disc(data)) / (torch.sum(disc(data) * (1-disc(data))) * 1) ** 0.5


def test_calibration_gaussian(gen, disc, num, numpy=False):

    train_data = gaussian_mix_generator(num)
    points = gen(generate_noise(num, gen.module.nz))

    data = torch.cat((train_data.to(0, torch.float), points.to(0, torch.float)), dim=0)
    labels = torch.cat((torch.full((num, 1), 1).cuda(), torch.full((num, 1), 0).cuda()), dim=0)

    if numpy:
        labels=labels.cpu().detach().numpy().reshape(-1)
    return z_calibration_stat(disc, data, labels, numpy)


def plot_calibration_curve(gen, disc, num, numpy=False):

    train_data = gaussian_mix_generator(num)
    points = gen(generate_noise(num // 1, gen.module.nz))

    data = torch.cat((train_data.to(0, torch.float), points.to(0, torch.float)), dim=0)
    labels = torch.cat((torch.full((num, 1), 1).cuda(), torch.full((num // 1, 1), 0).cuda()), dim=0)

    predictions = disc(data)
    if not numpy:
        predictions = predictions.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    true_prob, mean_predicted_prob = calibration_curve(labels, predictions, n_bins=10)
    plt.plot(mean_predicted_prob, true_prob, "s-")
    plt.plot([0,1],[0,1], "k:")
    plt.ylabel("True fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title("Calibration Plot")
    plt.show()

def l2norm(p1, p2):
    return np.sum((p1 - p2) ** 2) ** 0.5

def std_from_nearest_mode(point, std=0.05, mu=[-2,-1,0,1,2]):
    cur = l2norm(point, np.array([mu[0], mu[0]]))
    nearest_mode = mu[0] * 10 + mu[0]
    for x in mu:
        for y in mu:
            new = l2norm(point, np.array([x, y]))
            if new <= cur:
                cur = new
                nearest_mode = x * 10 + y
    return cur / std, nearest_mode


def gaussian_metric(points, std=0.05, mu=[-2,-1,0,1,2]):
    total_std = 0
    num_high_quality = 0
    nearest_mode_set = set()

    for point in points:
        z, nearest_mode = std_from_nearest_mode(point, std, mu)
        # print(z)
        if z <= 4:
            num_high_quality += 1.
            nearest_mode_set.add(nearest_mode)

        total_std += z

    return len(nearest_mode_set), num_high_quality / len(points), total_std / len(points)


def jsd(points, std=0.05, mu=[-2,-1,0,1,2]):
    num_high_quality = 0.
    sample_frequencies = {x * 10 + y : 0 for x in mu for y in mu}
    support = list(sample_frequencies.keys())
    for point in points:
        z, nearest_mode = std_from_nearest_mode(point, std, mu)
        if z <= 4:
            num_high_quality += 1
            sample_frequencies[nearest_mode] += 1

    sample_probabilities = {k: v / len(points) for k, v in sample_frequencies.items()}
    # print(sample_probabilities)
    uniform_probabilities = {k: 0.04 for k, v in sample_frequencies.items()}

    KL_s = 0
    KL_u = 0
    for x in support:
        s_x = sample_probabilities[x]
        u_x = uniform_probabilities[x]
        denom = (s_x + u_x) / 2
        KL_s += 0 if s_x == 0 else np.log(s_x / denom) * s_x
        KL_u += 0 if u_x == 0 else np.log(u_x / denom) * u_x

    return (1/2.) * (KL_s + KL_u)



#
# test = gaussian_mix_generator(10000).cpu().detach().numpy()
# print(gaussian_metric(test))
# print(jsd(test))
