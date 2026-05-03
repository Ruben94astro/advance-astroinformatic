import time
import os
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')   # ← descomenta esta línea (importante para evitar ventanas)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ------------------------------------------------------------
# 1. Función que grafica los tres paneles
# ------------------------------------------------------------
def plot_three(name_lc_raw, name_lc_outliers, name_lc_cbv):
    ''' function that takes the lc information for plotting '''
    # Raw
    lc_raw = pd.read_csv(name_lc_raw, sep='\s+', names=['JD', 'mag', 'err'])
    lc_raw = lc_raw.apply(pd.to_numeric, errors='coerce').dropna()

    # Outliers
    lc_outliers = pd.read_csv(name_lc_outliers, sep=',', names=['JD', 'mag', 'err'])
    lc_outliers = lc_outliers.apply(pd.to_numeric, errors='coerce').dropna()

    # Median after CBV
    lc_cbv = pd.read_csv(name_lc_cbv, sep='\s+',
                         names=['JD', 'mag_clean', 'mag_after_cbv', 'err'])
    lc_cbv = lc_cbv.apply(pd.to_numeric, errors='coerce').dropna()

    folder_name = os.path.basename(os.path.dirname(name_lc_raw))
    base_name = os.path.basename(name_lc_raw)
    without_ext = os.path.splitext(base_name)[0]
    parts = without_ext.split('_')
    name_obj = parts[0]
    sector = '_'.join(parts[1:])
    name_pdf = f"TESS_light_curve_{name_obj}_{sector}_{folder_name}.pdf"

    # Estilo
    sns.set_theme(style="ticks", context="paper", palette="viridis")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                                        gridspec_kw={'height_ratios': [1, 1, 1]})

    # Panel 1: Raw
    if not lc_raw.empty:
        ax1.errorbar(lc_raw['JD'], lc_raw['mag'], yerr=lc_raw['err'],
                     fmt='none', ecolor='green', alpha=0.1,
                     capsize=0, zorder=1, label='Error')
        sns.scatterplot(data=lc_raw, x='JD', y='mag', hue='mag',
                        palette='viridis', s=5, legend=False, ax=ax1)
        ax1.invert_yaxis()
        ax1.set_ylabel('Magnitude')
        ax1.set_title(f'TESS {name_obj} {folder_name} - Raw')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='lower left', framealpha=0.5)
    else:
        ax1.text(0.5, 0.5, 'No valid data', transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title(f'TESS {name_obj} {folder_name} - Raw (no data)')
    sns.despine(top=True, right=True, ax=ax1)

    # Panel 2: Outlier cleaned
    if not lc_outliers.empty:
        ax2.errorbar(lc_outliers['JD'], lc_outliers['mag'], yerr=lc_outliers['err'],
                     fmt='none', ecolor='green', alpha=0.1,
                     capsize=0, zorder=1, label='Error')
        sns.scatterplot(data=lc_outliers, x='JD', y='mag', hue='mag',
                        palette='viridis', s=5, legend=False, ax=ax2)
        ax2.invert_yaxis()
        ax2.set_ylabel('Magnitude')
        ax2.set_title(f'TESS {name_obj} {folder_name} - Outlier cleaned')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc='lower left', framealpha=0.5)
    else:
        ax2.text(0.5, 0.5, 'No valid data', transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title(f'TESS {name_obj} {folder_name} - Outlier cleaned (no data)')
    sns.despine(top=True, right=True, ax=ax2)

    # Panel 3: Median after CBV
    if not lc_cbv.empty:
        ax3.errorbar(lc_cbv['JD'], lc_cbv['mag_after_cbv'], yerr=lc_cbv['err'],
                     fmt='none', ecolor='green', alpha=0.1,
                     capsize=0, zorder=1, label='Error')
        sns.scatterplot(data=lc_cbv, x='JD', y='mag_after_cbv', hue='mag_after_cbv',
                        palette='viridis', s=5, legend=False, ax=ax3)
        ax3.invert_yaxis()
        ax3.set_ylabel('Magnitude')
        ax3.set_title(f'TESS {name_obj} {folder_name} - Median after CBV')
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.legend(loc='lower left', framealpha=0.5)
    else:
        ax3.text(0.5, 0.5, 'No valid data', transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title(f'TESS {name_obj} {folder_name} - Median after CBV (no data)')
    sns.despine(top=True, right=True, ax=ax3)

    ax3.set_xlabel('Julian Date')
    plt.tight_layout()
    # Guardar en la misma carpeta del raw
    output_dir = os.path.dirname(name_lc_raw)
    pdf_path = os.path.join(output_dir, name_pdf)
    plt.savefig(name_pdf, dpi=300)
    plt.close()
    # print(f"Saved: {pdf_path}")

# ------------------------------------------------------------
# 2. Recolectar tripletas válidas
# ------------------------------------------------------------
def collect_triplets(raw_root, outlier_root, median_root,
                     outlier_suffix='_cleaned',
                     median_subfolder_prefix='_lc_median_after_cbv_detrended_'):
    triplets = []
    for root, dirs, files in os.walk(raw_root):
        for file in files:
            if not file.endswith('.lc'):
                continue
            raw_path = os.path.join(root, file)
            base = os.path.splitext(file)[0]
            rel_path = os.path.relpath(root, raw_root)
            if rel_path == '.':
                rel_path = ''

            # Outlier
            outlier_name = base + outlier_suffix + '.lc'
            outlier_path = os.path.join(outlier_root, rel_path, outlier_name)
            if not os.path.exists(outlier_path):
                continue

            # Median
            if rel_path:
                median_subfolder = median_subfolder_prefix + rel_path
            else:
                median_subfolder = ''
            median_path = os.path.join(median_root, median_subfolder, file)
            if not os.path.exists(median_path):
                continue

            triplets.append((raw_path, outlier_path, median_path))
    return triplets

# ------------------------------------------------------------
# 3. Procesamiento en paralelo
# ------------------------------------------------------------
def process_all_triplets_parallel(raw_root, outlier_root, median_root,
                                  outlier_suffix='_cleaned',
                                  median_subfolder_prefix='_lc_median_after_cbv_detrended_',
                                  n_workers=None):
    print("Collecting valid triplets...")
    triplets = collect_triplets(raw_root, outlier_root, median_root,
                                outlier_suffix, median_subfolder_prefix)
    total = len(triplets)
    print(f"Found {total} valid triplets.")
    if total == 0:
        print("No triplets to process.")
        return

    if n_workers is None:
        n_workers = mp.cpu_count()
    print(f"Using {n_workers} worker processes.\n")

    start_time = time.time()
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(plot_three, triplets)
    elapsed = time.time() - start_time

    print("\n" + "="*50)
    print(f"SUMMARY")
    print(f"  Total triplets processed: {total}")
    print(f"  Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print("="*50)

# ------------------------------------------------------------
# 4. Ejecución principal
# ------------------------------------------------------------
if __name__ == "__main__":
    raw_folder = "_TESS_lightcurves_raw"
    outlier_folder = "_TESS_lightcurves_outliercleaned"
    median_folder = "_TESS_lightcurves_median_after_detrended"

    process_all_triplets_parallel(raw_folder, outlier_folder, median_folder,
                                  outlier_suffix='_cleaned',
                                  median_subfolder_prefix='_lc_median_after_cbv_detrended_')