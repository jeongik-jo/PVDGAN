import matplotlib.pyplot as plt
import os


def file_to_array(path):
    with open(path) as f:
        return [float(v) for v in f.readlines()]

def _graph(vae, dls, pvd, title, y_label, file_name, ylim=None):
    epochs = [i+1 for i in range(100)]
    plt.title(title)
    plt.plot(epochs, vae, label='VAEGAN')
    plt.plot(epochs, dls, label='DLSGAN')
    plt.plot(epochs, pvd, label='PVDGAN (ours)')
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    plt.savefig('./results/' + file_name + '.png', dpi=300)
    plt.clf()


dls_path = r'DLSGAN/results/figures'
pvd_path = r'PVDGAN/results/figures'
vae_path = r'VAEGAN/results/figures'

def draw_graphs():
    _graph(
        file_to_array(vae_path + r'\fid.txt'),
        file_to_array(dls_path + r'\fid.txt'),
        file_to_array(pvd_path + r'\fid.txt'),
        'Generative performance',
        'FID',
        'FID',
        (0, 40)
    )
    _graph(
        file_to_array(vae_path + r'\precision.txt'),
        file_to_array(dls_path + r'\precision.txt'),
        file_to_array(pvd_path + r'\precision.txt'),
        'Generative performance',
        'Precision',
        'Precision'
    )
    _graph(
        file_to_array(vae_path + r'\recall.txt'),
        file_to_array(dls_path + r'\recall.txt'),
        file_to_array(pvd_path + r'\recall.txt'),
        'Generative performance',
        'Recall',
        'Recall'
    )
    _graph(
        file_to_array(vae_path + r'\fake_psnr.txt'),
        file_to_array(dls_path + r'\fake_psnr.txt'),
        file_to_array(pvd_path + r'\fake_psnr.txt'),
        'Inversion performance',
        'Fake PSNR',
        'fake_psnr'
    )
    _graph(
        file_to_array(vae_path + r'\fake_ssim.txt'),
        file_to_array(dls_path + r'\fake_ssim.txt'),
        file_to_array(pvd_path + r'\fake_ssim.txt'),
        'Inversion performance',
        'Fake SSIM',
        'fake_ssim'
    )
    _graph(
        file_to_array(vae_path + r'\real_psnr.txt'),
        file_to_array(dls_path + r'\real_psnr.txt'),
        file_to_array(pvd_path + r'\real_psnr.txt'),
        'Comprehensive performance',
        'Real PSNR',
        'real_psnr'
    )
    _graph(
        file_to_array(vae_path + r'\real_ssim.txt'),
        file_to_array(dls_path + r'\real_ssim.txt'),
        file_to_array(pvd_path + r'\real_ssim.txt'),
        'Comprehensive performance',
        'Real SSIM',
        'real_ssim'
    )

draw_graphs()