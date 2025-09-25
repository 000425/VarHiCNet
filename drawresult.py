#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import logging
from datetime import datetime

# ==================== 初始化设置 ====================
# 禁用所有警告
warnings.filterwarnings("ignore")

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='plotting.log',
    filemode='w'
)
logger = logging.getLogger()

# ==================== Matplotlib强制设置 ====================
try:
    import matplotlib as mpl
    mpl.use('Agg')  # 必须在所有matplotlib导入前设置
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
except ImportError as e:
    logger.error(f"导入依赖失败: {str(e)}")
    sys.exit(1)

# ==================== 数据准备 ====================
def prepare_data():
    """准备绘图数据"""
    data = {
        'Method': ['HISV']*6 + ['HINT-TL']*3 + ['HiSVision']*6 + ['Hic_break finder']*6 + ['VarHiCNet']*6,
        'Sample': ['K562', 'T47D', 'SK-N-MC']*8,
        'Type': ['inter']*3 + ['intra']*3 + ['inter']*3 + ['inter']*3 + ['intra']*3 + ['inter']*3 + ['intra']*3,
        'Recall': [0.1818, 0.4137, 0.8888, 0.5000, 0.4347, 0.8888,
                   0.2727, 0.2758, 0.2222,
                   0.3333, 0.5862, 1, 0.2500, 0.3636, 0.6250,
                   0.4240, 0.6200, 1, 0.4583, 0.5454, 0.625,
                   0.3939, 0.5517, 1, 0.4583, 0.5000, 0.6250],
        'Precision': [0.3333, 0.9230, 0.421, 0.3750, 0.7142, 0.421,
                      0.2195, 0.1702, 0.6666,
                      0.6111, 0.8947, 1, 0.4000, 0.6428, 0.8333,
                      0.3589, 0.5800, 0.9, 0.4780, 0.7058, 0.8333,
                      0.6500, 0.7619, 1, 0.5882, 0.8462, 0.8333],
        'F1': [0.2352, 0.5713, 0.5713, 0.4285, 0.5404, 0.5713,
               0.2432, 0.2104, 0.3333,
               0.4314, 0.7083, 1, 0.3076, 0.4864, 0.7142,
               0.3880, 0.5990, 0.9473, 0.4679, 0.6499, 0.7142,
               0.4889, 0.6429, 1, 0.5152, 0.6284, 0.7142]
    }
    return pd.DataFrame(data)

# ==================== 绘图函数 ====================
def create_plot(df, output_path):
    """创建并保存图表"""
    try:
        # 创建画布
        fig, ax = plt.subplots(figsize=(14, 10))
        plt.style.use('seaborn-v0_8')

        # 绘制F1等值线
        x = y = np.linspace(0.01, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = 2 * X * Y / (X + Y)
        contour = ax.contour(X, Y, Z, levels=[0.2, 0.4, 0.6, 0.8], 
                           colors='gray', linestyles=':', alpha=0.7)
        ax.clabel(contour, inline=True, fontsize=10, fmt='%.1f')

        # 定义样式
        style_config = {
            'colors': {
                'HISV': '#1f77b4',
                'HINT-TL': '#ff7f0e',
                'HiSVision': '#2ca02c',
                'Hic_break finder': '#d62728',
                'VarHiCNet': '#9467bd'
            },
            'markers': {
                'K562': 'o',
                'T47D': 's',
                'SK-N-MC': 'D'
            },
            'edgecolors': {
                'inter': 'white',
                'intra': 'black'
            }
        }

        # 绘制散点
        for method in df['Method'].unique():
            for sample in df['Sample'].unique():
                subset = df[(df['Method'] == method) & (df['Sample'] == sample)]
                if not subset.empty:
                    for _, row in subset.iterrows():
                        ax.scatter(
                            row['Precision'], 
                            row['Recall'],
                            s=600 * row['F1'],
                            c=style_config['colors'][row['Method']],
                            marker=style_config['markers'][row['Sample']],
                            alpha=0.9,
                            edgecolor=style_config['edgecolors'][row['Type']],
                            linewidth=1.5,
                            zorder=10
                        )

        # 添加标签
        ax.set_xlabel('Precision', fontsize=16)
        ax.set_ylabel('Recall', fontsize=16)
        ax.set_title('Precision-Recall Performance Comparison\n(Point Size ∝ F1 Score)', 
                    fontsize=18, pad=20)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)

        # 创建图例
        legend_elements = [
            *[Line2D([0], [0], marker='o', color='w', label=method,
                    markerfacecolor=color, markersize=12) 
             for method, color in style_config['colors'].items()],
            *[Line2D([0], [0], marker=marker, color='w', label=sample,
                    markerfacecolor='gray', markersize=12) 
             for sample, marker in style_config['markers'].items()],
            *[Line2D([0], [0], marker='o', color='k', label=f'Type: {typ}',
                    markerfacecolor='lightgray', markeredgecolor=ec, markersize=12) 
             for typ, ec in style_config['edgecolors'].items()],
            Line2D([0], [0], linestyle=':', color='gray', label='F1 Contour')
        ]

        ax.legend(
            handles=legend_elements, 
            bbox_to_anchor=(1.05, 1), 
            loc='upper left',
            framealpha=1,
            fontsize=12,
            title='Legend:',
            title_fontsize=13
        )

        # 保存图像
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 验证文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            logger.info(f"成功保存图像到 {output_path} (大小: {file_size:.1f}KB)")
            return True
        else:
            logger.error("文件保存失败")
            return False

    except Exception as e:
        logger.error(f"绘图过程中发生错误: {str(e)}", exc_info=True)
        return False

# ==================== 主程序 ====================
if __name__ == "__main__":
    logger.info("程序启动")
    
    # 准备输出路径
    output_file = os.path.abspath('precision_recall_f1_plot.jpg')
    logger.info(f"输出文件路径: {output_file}")
    
    try:
        # 准备数据
        df = prepare_data()
        logger.info("数据准备完成")
        
        # 创建并保存图表
        success = create_plot(df, output_file)
        
        if success:
            logger.info("程序执行成功")
            sys.exit(0)
        else:
            logger.error("程序执行失败")
            sys.exit(1)
            
    except Exception as e:
        logger.critical(f"程序崩溃: {str(e)}", exc_info=True)
        sys.exit(1)