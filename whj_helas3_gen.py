import cooltools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import cooler
import cooltools.lib.plotting
from matplotlib.colors import LogNorm

cool_file = '/public_data2/dongchengyao/HCC1954.mcool'
cell_name = cool_file.split("/")[-1].split('.')[0]


def Calculating_diagonal_data(matrix):
    """
    # normalization matrix by diagonal to remove distance bias
    Calculating each diagonal mean and std
    """
    N, M = len(matrix), len(matrix[0])
    Diagonal_mean = np.full(min(N, M), 0.0)  # Ensure the array size is min(N, M)
    Diagonal_std = np.full(min(N, M), 0.0)  # Ensure the array size is min(N, M)
    
    for d in range(min(N, M)):  # Loop limit changed to min(N, M)
        intermediate = []
        c = d
        r = 0
        while r < N and c < M:  # Ensure we do not go beyond the matrix dimensions
            intermediate.append(matrix[r][c])
            r += 1
            c += 1
        intermediate = np.array(intermediate)
        Diagonal_mean[d] = np.mean(intermediate) if len(intermediate) > 0 else 0
        Diagonal_std[d] = np.std(intermediate) if len(intermediate) > 0 else 0
    
    return Diagonal_mean, Diagonal_std






def Distance_normalization(matrix):
    """
    # normalization matrix by diagonal to remove distance bias
    norm_data = (data - mean_data) / mean_std
    """

    Diagonal_mean, Diagonal_std = Calculating_diagonal_data(matrix)
    N, M = len(matrix), len(matrix[0])

    for d in range(min(N, M)):  # Loop limit changed to min(N, M)
        c = d
        r = 0
        while r < N and c < M:  # Ensure r and c do not go out of bounds
            if c < M and r < N:  # Added a check to make sure we are within bounds
                if Diagonal_std[d] == 0:
                    matrix[r][c] = 0
                else:
                    # Ensure we don't subtract an out-of-bound mean
                    if matrix[r][c] - Diagonal_mean[d] < 0:
                        matrix[r][c] = 0
                    else:
                        matrix[r][c] = (matrix[r][c] - Diagonal_mean[d]) / Diagonal_std[d]
            r += 1
            c += 1
    return matrix



def generate_Sv_iamge(cool_file_path, resolution, chr1, chr2, pos1, pos2, inter_bins, intra_bins, label=None):
    m = [0.3, 0.6, 1, 1.3, 1.6]
    n = [0.3, 0.6, 1, 1.3, 1.6]
    num = 0
    if chr1 != chr2:
        clr = cooler.Cooler(f'{cool_file_path}::resolutions/{resolution}')
        contact_matrix = clr.matrix(balance=False)
        SV_region = contact_matrix.fetch(chr1, chr2)

        SV_region = Distance_normalization(SV_region)
        for i in m:
            for j in n:
                num += 1
                left = int((pos1 / resolution) - i * inter_bins)
                right = int((pos1 / resolution) + (2 - i) * inter_bins)
                up = int((pos2 / resolution) - j * inter_bins)
                bottom = int((pos2 / resolution) + (2 - j) * inter_bins)

                if left < 0 or up < 0:
                    break
                sv_subregion = SV_region[left:right, up:bottom]

                # norm = LogNorm(vmin=0, vmax=100)
                fruitpunch = sns.blend_palette(['white', 'red'], as_cmap=True)
                plt.figure(figsize=(sv_subregion.shape[0], sv_subregion.shape[1]), dpi=resolution / 5000)
                plt.imshow(sv_subregion, cmap=fruitpunch)
                plt.axis('off')
                fig = plt.gcf()
                fig.set_size_inches(sv_subregion.shape[0], sv_subregion.shape[1], forward=True)
                fig.tight_layout()
                # 修改后的文件保存路径
                folder_path = f'./{cell_name}_SV_image_whj111_{resolution}/{chr1}_{pos1}:{pos2}_{label}_{resolution}_{num}.jpg'

                # 确保目录存在，如果不存在则创建
                directory = os.path.dirname(folder_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # 保存图片
                plt.savefig(folder_path, dpi=resolution / 5000)

                plt.close()
                print(f'./{cell_name}_SV_image_{resolution}/{chr1}:{chr2},{pos1}:{pos2}_{num}.jpg  创建成功')
    else:
        if pos2 - pos1 > 1000000:
            clr = cooler.Cooler(f'{cool_file_path}::resolutions/{resolution}')
            contact_matrix = clr.matrix(balance=False)
            SV_region = contact_matrix.fetch(chr1)
            for i in m:
                for j in n:
                    num += 1
                    left = int((pos1 / resolution) - i * intra_bins)
                    right = int((pos1 / resolution) + (2 - i) * intra_bins)
                    up = int((pos2 / resolution) - j * intra_bins)
                    bottom = int((pos2 / resolution) + (2 - j) * intra_bins)

                    if left < 0 or up < 0:
                        break
                    sv_subregion = SV_region[left:right, up:bottom]

                    # norm = LogNorm(vmin=0, vmax=100)
                    fruitpunch = sns.blend_palette(['white', 'red'], as_cmap=True)
                    plt.figure(figsize=(sv_subregion.shape[0], sv_subregion.shape[1]), dpi=resolution / 5000)
                    plt.imshow(sv_subregion, cmap=fruitpunch)
                    plt.axis('off')
                    fig = plt.gcf()
                    fig.set_size_inches(sv_subregion.shape[0], sv_subregion.shape[1], forward=True)
                    fig.tight_layout()
                    # 修改后的文件保存路径
                    folder_path = f'./{cell_name}_SV_image_whj111_{resolution}/{chr1}_{pos1}:{pos2}_{label}_{resolution}_{num}.jpg'

                    # 确保目录存在，如果不存在则创建
                    directory = os.path.dirname(folder_path)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # 保存图片
                    plt.savefig(folder_path, dpi=resolution / 5000)

                    plt.close()
                    print(f'./{cell_name}_SV_image_{resolution}/{chr1}_{pos1}:{pos2}_{num}.jpg  创建成功')


SVs = pd.read_csv('/public_data2/dongchengyao/hcc1954_svlist.csv')

for row in SVs.index:
    for res in [25000, 50000, 100000]:
        folder = f'./{cell_name}_SV_image_{res}'
        if not os.path.exists(folder):
            os.mkdir(folder)
        inter_bins = int((25000 / res) * 80)
        intra_bins = int((25000 / res) * 80)
        generate_Sv_iamge(cool_file, res, SVs.loc[row]['chrom1'], SVs.loc[row]['chrom2']
                          , SVs.loc[row]['breakpoint1'], SVs.loc[row]['breakpoint2'], inter_bins, intra_bins,
                          label=SVs.loc[row]['strands'])