conda activate r_env

conda install -y --file r-requirements.txt

conda install -y -c conda-forge r-ggplotify r-ggnewscale

Rscript install_r_packages.R
