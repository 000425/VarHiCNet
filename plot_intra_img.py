import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import cooler
from numpy import errstate, isneginf, array
import argparse
from scipy.ndimage import uniform_filter

def calculate_oe(matrix, bandwidth):
    """带状区域OE矩阵计算"""
    n = matrix.shape[0]
    expected = np.zeros(n)

    # 计算每个距离的期望值
    for d in range(1, n):
        diag = np.diag(matrix, d)
        valid = diag[diag > 0]
        if len(valid) >= 5:  # 最小样本量阈值
            expected[d] = np.mean(valid)

    # 平滑期望值曲线
    expected = uniform_filter(expected, size=5, mode='mirror')

    # 生成OE矩阵（仅处理带状区域）
    oe_matrix = np.zeros_like(matrix)
    for i in range(n):
        # 仅在带状范围内计算
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            d = abs(i - j)
            # 如果该距离没有期望值或距离为0则跳过
            if d == 0 or expected[d] == 0:
                continue
            oe_matrix[i, j] = np.log1p(matrix[i, j] / expected[d])

    return oe_matrix

def Calculating_diagonal_data(matrix):
    """
    # normalization matrix by diagonal to remove distance bias
    Calculating each diagonal mean and std
    """
    N, M = len(matrix), len(matrix[0])
    Diagonal_mean = np.full(min(N, M), 0.0)  # Ensure the array size is min(N, M)
    Diagonal_std = np.full(min(N, M), 0.0)   # Ensure the array size is min(N, M)

    for d in range(min(N, M)):
        intermediate = []
        c = d
        r = 0
        while r < N and c < M:
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

    for d in range(min(N, M)):
        c = d
        r = 0
        while r < N and c < M:
            if c < M and r < N:
                if Diagonal_std[d] == 0:
                    matrix[r][c] = 0
                else:
                    if matrix[r][c] - Diagonal_mean[d] < 0:
                        matrix[r][c] = 0
                    else:
                        matrix[r][c] = (matrix[r][c] - Diagonal_mean[d]) / Diagonal_std[d]
            r += 1
            c += 1
    return matrix

def hybrid_normalization(matrix, bandwidth):
    """
    染色体内混合归一化：
      - 带状区域使用OE（calculate_oe）
      - 带状区域外使用Z-score（Distance_normalization）
    """
    # 根据传入的 bandwidth 来计算
    oe_part = calculate_oe(matrix.copy(), bandwidth)
    zscore_part = Distance_normalization(matrix.copy())
    
    # 创建带状掩膜，仅在带宽范围内使用 OE，其他位置使用 Z-score
    n = matrix.shape[0]
    mask = np.zeros_like(matrix, dtype=bool)
    for i in range(n):
        start = max(0, i - bandwidth)
        end = min(n, i + bandwidth + 1)
        mask[i, start:end] = True
    
    return np.where(mask, oe_part, zscore_part)

def main():
    # 这里指定好你的 .mcool 文件
    cool_file = "/public_data2/dongchengyao/T47D.mcool"
    cell_name = cool_file.split("/")[-1].split('.')[0]
    resolution = 50000

    # 从 .mcool 文件中按指定分辨率读取 contact_matrix
    clr = cooler.Cooler(f'{cool_file}::resolutions/{resolution}')
    contact_matrix = clr.matrix(balance=False)

    chromosomes = [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6',
        'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
        'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18',
        'chr19', 'chr20', 'chr21', 'chr22', 'chrX'
    ]

    # 你可以根据需要手动修改以下两个参数
    tile_size = 160
    overlap_percentage = 0.2
    overlap = int(tile_size * overlap_percentage)
    
    # 直接手动设置带宽（可根据需要自行修改数值）
    bandwidth = 400

    # 指定保存输出图片的文件夹
    output_folder = "/public_data2/dongchengyao/2025_T47D_intra_50000_400"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for chrom in chromosomes:
        chr_mat = contact_matrix.fetch(chrom)
        # 使用手动设定的带宽进行混合归一化
        chr_mat = hybrid_normalization(chr_mat, bandwidth)
        
        height = chr_mat.shape[0]
        width = chr_mat.shape[1]
        print(chrom, width, height)

        x = 0
        y = 0
        row = 0

        while y < (height - 10):
            row += 1
            count = 0
            while x < (width - 10):
                count += 1
                if x != y:
                    sv_subregion = chr_mat[y:y+tile_size, x:x+tile_size]
                    
                    fruitpunch = sns.blend_palette(['white', 'red'], as_cmap=True)
                    plt.figure(figsize=(sv_subregion.shape[0], sv_subregion.shape[1]), dpi=5)
                    plt.imshow(sv_subregion, cmap=fruitpunch)
                    plt.axis('off')
                    
                    fig = plt.gcf()
                    fig.set_size_inches(sv_subregion.shape[0], sv_subregion.shape[1], forward=True)
                    fig.tight_layout()

                    out_filename = f"{chrom}_{row}_{count}.jpg"
                    out_path = os.path.join(output_folder, out_filename)

                    plt.savefig(out_path, pad_inches=0.0, bbox_inches='tight', dpi=5)
                    plt.close()
                x += tile_size - overlap
            x = 0
            y += tile_size - overlap
        
        print(f'{chrom} done')

if __name__ == '__main__':
    main()
