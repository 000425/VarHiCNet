import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import cooler

def Calculating_diagonal_data(matrix):
    """
    # normalization matrix by diagonal to remove distance bias
    Calculating each diagonal mean and std
    """
    N, M = len(matrix), len(matrix[0])
    Diagonal_mean = np.full(min(N, M), 0.0)  # Ensure the array size is min(N, M)
    Diagonal_std = np.full(min(N, M), 0.0)   # Ensure the array size is min(N, M)

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
            if c < M and r < N:  # 检查下标是否越界
                if Diagonal_std[d] == 0:
                    matrix[r][c] = 0
                else:
                    # 确保不减去越界的mean
                    if matrix[r][c] - Diagonal_mean[d] < 0:
                        matrix[r][c] = 0
                    else:
                        matrix[r][c] = (matrix[r][c] - Diagonal_mean[d]) / Diagonal_std[d]
            r += 1
            c += 1
    return matrix

def main():
    # 你要读取的 mcool 文件路径
    cool_file = "/public_data2/wangtao/HISV/K562/K562.mcool"
    print(cool_file)
    cell_name = cool_file.split("/")[-1].split('.')[0]
    print(cell_name)

    # 你想要使用的分辨率
    resolution = 50000

    # 读取 cooler 文件
    clr = cooler.Cooler(f'{cool_file}::resolutions/{resolution}')
    contact_matrix = clr.matrix(balance=True)

    # 你要处理的染色体列表
    chr_list = [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5',
        'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
        'chr16', 'chr17', 'chr18', 'chr19', 'chr20',
        'chr21', 'chr22', 'chrX'
    ]

    # 在这里指定你想保存文件的目标文件夹
    output_folder = "/public_data2/dongchengyao/2025_K562_inter_50000"  # 修改为你想输出的路径
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tile_size = 200
    overlap_percentage = 0.2
    overlap = int(tile_size * overlap_percentage)

    # 遍历所有的染色体对（i < j）
    for i in range(len(chr_list)):
        for j in range(i + 1, len(chr_list)):
            print(chr_list[i], chr_list[j])
            # 从 contact matrix 中获取这两个染色体的矩阵
            chr_mat = contact_matrix.fetch(chr_list[i], chr_list[j])

            # 距离归一化
            chr_mat = Distance_normalization(chr_mat)

            height = chr_mat.shape[0]
            width = chr_mat.shape[1]
            print(width, height)

            x = 0
            y = 0
            row = 0

            # 按照 tile_size + overlap 的方式去遍历矩阵
            while y < height - 10:
                row += 1
                count = 0
                while x < width - 10:
                    count += 1
                    sv_subregion = chr_mat[y:y+tile_size, x:x+tile_size]

                    # 如果感兴趣区域中大于 0.001 的值超过一定数量，就输出图片
                    if np.sum(sv_subregion > 0.001) > 3:
                        fruitpunch = sns.blend_palette(['white', 'red'], as_cmap=True)
                        plt.figure(figsize=(sv_subregion.shape[0], sv_subregion.shape[1]), dpi=4)
                        plt.imshow(sv_subregion, cmap=fruitpunch)
                        plt.axis('off')

                        fig = plt.gcf()
                        fig.set_size_inches(sv_subregion.shape[0], sv_subregion.shape[1], forward=True)
                        fig.tight_layout()

                        # 将文件保存在我们指定的文件夹中
                        out_filename = f"{chr_list[i]}_{chr_list[j]}_{row}_{count}.jpg"
                        out_path = os.path.join(output_folder, out_filename)
                        plt.savefig(out_path, pad_inches=0.0, bbox_inches='tight', dpi=4)
                        plt.close()

                    x += tile_size - overlap
                x = 0
                y += tile_size - overlap

if __name__ == '__main__':
    main()
