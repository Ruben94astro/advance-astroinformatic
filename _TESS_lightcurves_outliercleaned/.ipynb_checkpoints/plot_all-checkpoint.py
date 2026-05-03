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


#first we crete a functions for plot one lc

def plot_lc(name_lc):

    ''' function that take the name of the lc and give a plot in pdf 
    only works on lc that have 3 columns'''
    lc = pd.read_csv(name_lc, sep=',',
                    names = ['JD', 'mag', 'err'])
    lc = lc.apply(pd.to_numeric, errors='coerce').dropna()
    if lc.empty:
        print(f"Sin datos válidos: {name_lc}")
        return

    ##adding the folder the lightcurves below
    folder_name = os.path.basename(os.path.dirname(name_lc))
    
    #taking out the extension
    without_ext = os.path.splitext(name_lc)[0]
    #configuration for naming the pdf
     #---> name of the folder
    base_name = os.path.basename(name_lc) # '41259805_sector01_4_2.lc'
    without_ext = os.path.splitext(base_name)[0] # ---->>>> whitout lv'41259805_sector01_4_2'
    parts = without_ext.split('_')
    name_obj = parts[0]   # '41259805'
    sector = '_'.join(parts[1:])  #sector '01_4_2' 
    name_pdf = f"TESS {name_obj}_{sector}_{folder_name}.pdf"

   # saving in the same folder .lc -----
    output_dir = os.path.dirname(name_lc)   # original path
    pdf_path = os.path.join(output_dir, name_pdf)
    
    # using seaborn for aestetic reason
    sns.set_theme(style="ticks", context="paper", palette="viridis")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot with error bars (using matplotlib errorbar)
    # fmt='b.' means blue points, markersize controls point size
    # capsize=0 removes the caps at the ends of error bars (cleaner look)
    ax.errorbar(lc['JD'], lc['mag'], yerr=lc['err'], 
                fmt='none', ecolor='green', alpha=0.1, 
                capsize=0, zorder=1,label='Error (mag)')
    
    sns.scatterplot(data=lc, x='JD', y='mag', hue='mag', 
                    palette='viridis', s=5, legend=False,ax=ax)    
    
    # invert y axis
    ax.invert_yaxis()
    ax.set_xlabel(f'Julian Date', fontsize=12, labelpad=10)
    ax.set_ylabel(f'Magnitude', fontsize=12, labelpad=10)
    ax.set_title(f'TESS {name_obj} {folder_name}', fontsize=14, pad=15)
    
    # grid
    #ax.grid(True, linestyle='--', alpha=0.5, axis='both')
    #ax.legend(loc='best') 
    ax.legend(loc='lower left', framealpha=0.5)
    # more aestetic i guess...
    sns.despine(top=True, right=True)
    
    # plot and save pdf
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