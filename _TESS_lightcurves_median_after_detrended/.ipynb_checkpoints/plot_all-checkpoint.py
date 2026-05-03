import os
import glob
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import gc



def plot_lc(name_lc):
    ''' function that take the name of the lc and give a plot in pdf '''
    # Leer con los nombres reales de las columnas
    lc = pd.read_csv(name_lc, sep='\s+',
                     names=['JD', 'mag_clean', 'mag_after_cbv', 'err'])
    lc = lc.apply(pd.to_numeric, errors='coerce').dropna()

    # Elegir la curva a graficar (por ejemplo mag_clean)
    y_col = 'mag_clean'
    y_label = 'Magnitude (detrended clean)'

    folder_name = os.path.basename(os.path.dirname(name_lc))
    base_name = os.path.basename(name_lc)
    without_ext = os.path.splitext(base_name)[0]
    parts = without_ext.split('_')
    name_obj = parts[0]
    sector = '_'.join(parts[1:])
    name_pdf = f"TESS_{name_obj}_{sector}_{folder_name}.pdf"
    output_dir = os.path.dirname(name_lc)
    pdf_path = os.path.join(output_dir, name_pdf)

    sns.set_theme(style="ticks", context="paper", palette="viridis")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    fig, ax = plt.subplots(figsize=(10, 5))

    # Barras de error
    ax.errorbar(lc['JD'], lc[y_col], yerr=lc['err'],
                fmt='none', ecolor='green', alpha=0.1,
                capsize=0, zorder=1, label='Error bar')

    # Puntos coloreados por magnitud (usando la misma columna)
    sc = ax.scatter(lc['JD'], lc[y_col], c=lc[y_col],
                    cmap='viridis', s=5, zorder=2)

    ax.invert_yaxis()
    ax.set_xlabel('Julian Date', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'TESS {name_obj} {folder_name}', fontsize=14)
    ax.legend(loc='lower left', framealpha=0.5)
    sns.despine(top=True, right=True)

    plt.tight_layout()
    plt.savefig(pdf_path, dpi=300)
    plt.close()
    gc.collect()


def process_all_lc_parallel(root_folder, n_workers=None):
    # Recoger todos los archivos .lc primero
    lc_files = []
    for root, dirs, files in os.walk(root_folder):
        for f in files:
            if f.endswith('.lc'):
                lc_files.append(os.path.join(root, f))
    
    if n_workers is None:
        n_workers = mp.cpu_count()  # usar todos los núcleos disponibles
    
    with mp.Pool(processes=n_workers) as pool:
        pool.map(plot_lc, lc_files)


if __name__ == "__main__":
    root_folder = os.getcwd()
    process_all_lc_parallel(root_folder)