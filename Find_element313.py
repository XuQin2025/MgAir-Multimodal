import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格，全英文标签避免乱码
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']


def get_alloy_name(row, elements):
    """根据非零元素自动生成合金牌号，例如 Mg-3Al-1Zn"""
    comps = []
    for el in elements:
        val = row.get(el, 0)
        if pd.notna(val) and val > 1e-6:  # 忽略极小值或0
            val_str = f"{val:g}"  # 去除多余的小数点
            comps.append(f"{el}{val_str}")

    if comps:
        return "Mg-" + "-".join(comps)
    else:
        return "Pure Mg"


def screen_top5_normalized_performance():
    # 1. 读取数据
    file_path = 'rawdata.xlsx'
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
    except FileNotFoundError:
        print(f"Error: 找不到文件 {file_path}")
        return

    # 清洗列名
    df.columns = df.columns.str.strip()

    # 识别关键列
    col_voltage = 'Average discharge voltage/mV'
    col_efficiency = 'Anodic efficiency/%'

    possible_current_cols = [c for c in df.columns if "current density" in c.lower()]
    if not possible_current_cols:
        print("Error: 找不到 'current density' 列")
        return
    col_current = possible_current_cols[0]

    # 定义所有的合金元素列
    element_cols_all = [
        'Al', 'Ca', 'Mn', 'Zn', 'Sn', 'La', 'Ce', 'Er', 'In', 'Y',
        'Ga', 'Gd', 'Sm', 'Ge', 'Nd', 'Li', 'Sr', 'Zr', 'Fe', 'Cu',
        'Ni', 'Pb', 'Bi', 'Ag', 'Si', 'Co', 'Ti', 'Cd'
    ]
    # 仅保留数据表中真实存在的元素列
    valid_elements = [col for col in element_cols_all if col in df.columns]

    # --- 预处理：计算总含量和生成合金牌号 ---
    df['Total_Alloy_Content_wt%'] = df[valid_elements].sum(axis=1)
    df['Alloy_Composition'] = df.apply(lambda row: get_alloy_name(row, valid_elements), axis=1)

    # 目标电流密度
    target_currents = [2.5, 10, 40]

    # 存储所有高性能数据以便后续导出
    all_high_perf_data = []

    for current_val in target_currents:
        print(f"\n--- Analyzing Current Density: {current_val} mA/cm² ---")

        # 筛选特定电流密度数据 (使用容差0.1)
        mask = np.isclose(df[col_current], current_val, atol=0.1)
        df_sub = df[mask].copy()

        # 删除带有缺失值的目标列
        df_sub = df_sub.dropna(subset=[col_voltage, col_efficiency])

        if df_sub.empty:
            print(f"Warning: 数据中没有电流密度为 {current_val} 的有效记录，跳过。")
            continue

        print(f"Found {len(df_sub)} valid records.")

        # 2. Min-Max 归一化计算综合得分
        v_min, v_max = df_sub[col_voltage].min(), df_sub[col_voltage].max()
        e_min, e_max = df_sub[col_efficiency].min(), df_sub[col_efficiency].max()

        # 避免除以 0 的情况
        if v_max > v_min:
            df_sub['Norm_Voltage'] = (df_sub[col_voltage] - v_min) / (v_max - v_min)
        else:
            df_sub['Norm_Voltage'] = 0.0

        if e_max > e_min:
            df_sub['Norm_Efficiency'] = (df_sub[col_efficiency] - e_min) / (e_max - e_min)
        else:
            df_sub['Norm_Efficiency'] = 0.0

        # 综合评分 (最高2.0)
        df_sub['Combined_Score'] = df_sub['Norm_Voltage'] + df_sub['Norm_Efficiency']

        # 3. 提取 Top 5 高分数据
        # nlargest(5) 会自动帮我们找到评分最高的前五名
        top5_df = df_sub.nlargest(5, 'Combined_Score').copy()
        top5_df['Screened_Current_Density'] = current_val
        all_high_perf_data.append(top5_df)

        # 为了画图区分，做个标记
        df_sub['Is_Top5'] = False
        df_sub.loc[top5_df.index, 'Is_Top5'] = True

        # 4. 绘图
        plt.figure(figsize=(10, 8), dpi=150)

        # 绘制非 Top 5 常规点 (蓝色)
        sns.scatterplot(
            data=df_sub[~df_sub['Is_Top5']],
            x=col_voltage, y=col_efficiency,
            color='#4A90E2', alpha=0.7, s=80, edgecolor='w',
            label='Normal Performance'
        )

        # 绘制 Top 5 高性能点 (红色星星)
        if not top5_df.empty:
            sns.scatterplot(
                data=top5_df,
                x=col_voltage, y=col_efficiency,
                color='#D0021B', marker='*', s=350, edgecolor='k', linewidth=1.2,
                label='Top 5 (Highest Combined Score)', zorder=5
            )

            # 给 Top 5 点标上排名 (1, 2, 3, 4, 5)
            # 使用 iterrows 并进行 enumerate
            for rank, (idx, row) in enumerate(top5_df.iterrows(), 1):
                plt.annotate(
                    f"#{rank}",
                    (row[col_voltage], row[col_efficiency]),
                    xytext=(7, 7), textcoords='offset points',
                    fontsize=11, fontweight='bold', color='black',
                    bbox=dict(boxstyle="circle,pad=0.2", fc="yellow", ec="black", alpha=0.8)
                )

        # 美化图表
        plt.title(
            f'Performance Distribution @ {current_val} mA/cm$^2$\nHighlighted: Top 5 by Normalized Score (V + Eff)',
            fontsize=15, pad=15, fontweight='bold')
        plt.xlabel('Average Discharge Voltage (mV)', fontsize=13)
        plt.ylabel('Anodic Efficiency (%)', fontsize=13)

        # 动态扩大一些边界以容纳标签
        x_min, x_max = df_sub[col_voltage].min(), df_sub[col_voltage].max()
        y_min, y_max = df_sub[col_efficiency].min(), df_sub[col_efficiency].max()
        x_span = max(10, x_max - x_min)
        y_span = max(2, y_max - y_min)
        plt.xlim(x_min - x_span * 0.1, x_max + x_span * 0.1)
        plt.ylim(y_min - y_span * 0.1, y_max + y_span * 0.15)

        plt.legend(loc='lower left', frameon=True, shadow=True)
        plt.tight_layout()

        # 保存图片
        s_name = str(current_val).replace('.', '_')
        plot_name = f'top5_normalized_distribution_{s_name}mA.png'
        plt.savefig(plot_name, dpi=300)
        print(f"Saved plot: {plot_name}")
        plt.close()

    # 5. 数据统计与导出
    if all_high_perf_data:
        # 合并所有的 Top 5 数据
        final_top5_df = pd.concat(all_high_perf_data, ignore_index=True)

        # 统计频次与元素总含量
        summary_stats = final_top5_df.groupby(['Screened_Current_Density', 'Alloy_Composition']).agg(
            Frequency=('Alloy_Composition', 'count'),
            Mean_Total_Alloy_wt_pct=('Total_Alloy_Content_wt%', 'mean'),
            Mean_Combined_Score=('Combined_Score', 'mean'),
            Mean_Voltage=('Average discharge voltage/mV', 'mean'),
            Mean_Efficiency=('Anodic efficiency/%', 'mean')
        ).reset_index()

        # 按电流密度和频次降序排列
        summary_stats = summary_stats.sort_values(by=['Screened_Current_Density', 'Frequency'], ascending=[True, False])

        # 导出至新 Excel
        output_excel = 'top5_alloys_summary.xlsx'
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            # Sheet 1: 统计汇总表
            summary_stats.to_excel(writer, sheet_name='Frequency & Content Stats', index=False)
            # Sheet 2: 完整的 Top 5 原始数据点 (包括它们归一化前后的值)
            final_top5_df.to_excel(writer, sheet_name='Top 5 Raw Data', index=False)

        print(f"\nSuccess! All Top 5 data and statistics exported to: {output_excel}")
        print("\n--- Top 5 Alloys Frequency Preview ---")
        print(summary_stats.to_string(index=False))
    else:
        print("\nNo high performance alloys found across the specified current densities.")


if __name__ == "__main__":
    screen_top5_normalized_performance()