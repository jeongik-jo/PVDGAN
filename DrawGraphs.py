import matplotlib.pyplot as plt
import os


def file_to_array(path):
    with open(path) as f:
        return [float(v) for v in f.readlines()]


def _graph(mse, info, dls, pvd, vae_p0, vae_p1, vae_p10, title, y_label, file_name, ylim=None):
    epochs = [i+1 for i in range(100)]
    plt.title(title)
    plt.plot(epochs, mse, label='MSEGAN')
    plt.plot(epochs, info, label='InfoGAN')
    plt.plot(epochs, dls, label='DLSGAN')
    plt.plot(epochs, pvd, label='PVDGAN (ours)')
    plt.plot(epochs, vae_p0, label=r'VAEGAN $\lambda_{prr}=0$')
    plt.plot(epochs, vae_p1, label=r'VAEGAN $\lambda_{prr}=1$')
    plt.plot(epochs, vae_p10, label=r'VAEGAN $\lambda_{prr}=10$')
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)

    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/' + file_name + '.png', dpi=300)
    plt.clf()

dataset = r'FFHQ'
mse_path = dataset + r'/' + r'MSE/results/figures'
info_path = dataset + r'/' + r'InfoGAN/results/figures' 
dls_path = dataset + r'/' + r'DLSGAN/results/figures'
pvd_path = dataset + r'/' + r'PVDGAN/results/figures'
vae_p0_path = dataset + r'/' + r'VAEGAN_P0/results/figures'
vae_p1_path = dataset + r'/' + r'VAEGAN_P1/results/figures'
vae_p10_path = dataset + r'/' + r'VAEGAN_P10/results/figures'

def draw_graphs():
    f_name = '/fid.txt'
    _graph(
        file_to_array(mse_path + f_name),
        file_to_array(info_path + f_name),
        file_to_array(dls_path + f_name),
        file_to_array(pvd_path + f_name),
        file_to_array(vae_p0_path + f_name),
        file_to_array(vae_p1_path + f_name),
        file_to_array(vae_p10_path + f_name),
        dataset + ' Generative Performance',
        'FID',
        'fid',
        (0, 40)
    )
    f_name = '/precision.txt'
    _graph(
        file_to_array(mse_path + f_name),
        file_to_array(info_path + f_name),
        file_to_array(dls_path + f_name),
        file_to_array(pvd_path + f_name),
        file_to_array(vae_p0_path + f_name),
        file_to_array(vae_p1_path + f_name),
        file_to_array(vae_p10_path + f_name),
        dataset + ' Generative Performance',
        'Precision',
        'precision'
    )
    f_name = '/recall.txt'
    _graph(
        file_to_array(mse_path + f_name),
        file_to_array(info_path + f_name),
        file_to_array(dls_path + f_name),
        file_to_array(pvd_path + f_name),
        file_to_array(vae_p0_path + f_name),
        file_to_array(vae_p1_path + f_name),
        file_to_array(vae_p10_path + f_name),
        dataset + ' Generative Performance',
        'Recall',
        'recall'
    )
    f_name = '/fake_psnr.txt'
    _graph(
        file_to_array(mse_path + f_name),
        file_to_array(info_path + f_name),
        file_to_array(dls_path + f_name),
        file_to_array(pvd_path + f_name),
        file_to_array(vae_p0_path + f_name),
        file_to_array(vae_p1_path + f_name),
        file_to_array(vae_p10_path + f_name),
        dataset + ' Inversion Performance',
        'Fake PSNR',
        'fake_psnr'
    )
    f_name = '/fake_ssim.txt'
    _graph(
        file_to_array(mse_path + f_name),
        file_to_array(info_path + f_name),
        file_to_array(dls_path + f_name),
        file_to_array(pvd_path + f_name),
        file_to_array(vae_p0_path + f_name),
        file_to_array(vae_p1_path + f_name),
        file_to_array(vae_p10_path + f_name),
        dataset + ' Inversion Performance',
        'Fake SSIM',
        'fake_ssim'
    )
    f_name = '/real_psnr.txt'
    _graph(
        file_to_array(mse_path + f_name),
        file_to_array(info_path + f_name),
        file_to_array(dls_path + f_name),
        file_to_array(pvd_path + f_name),
        file_to_array(vae_p0_path + f_name),
        file_to_array(vae_p1_path + f_name),
        file_to_array(vae_p10_path + f_name),
        dataset + ' Inversion Performance',
        'Real PSNR',
        'real_psnr'
    )
    f_name = '/real_ssim.txt'
    _graph(
        file_to_array(mse_path + f_name),
        file_to_array(info_path + f_name),
        file_to_array(dls_path + f_name),
        file_to_array(pvd_path + f_name),
        file_to_array(vae_p0_path + f_name),
        file_to_array(vae_p1_path + f_name),
        file_to_array(vae_p10_path + f_name),
        dataset + ' Inversion Performance',
        'Real SSIM',
        'real_ssim'
    )

draw_graphs()