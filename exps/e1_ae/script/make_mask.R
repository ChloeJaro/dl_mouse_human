## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE,
                      include = FALSE,
                      cache.lazy = FALSE)


## ----libraries----------------------------------------------------------------
#Libraries
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(grid))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(pheatmap))
suppressPackageStartupMessages(library(ggplotify))
suppressPackageStartupMessages(library(ggnewscale))
suppressPackageStartupMessages(library(data.tree))
suppressPackageStartupMessages(library(RColorBrewer))


## ---Functions-----------------------------------------------------------------
source("/well/mars/users/uvy786/MouseHumanTranscriptomicSimilarity/functions/tree_tools.R")

## ---DATA----------------------------------------------------------------------
dfExprMouse <- as_tibble(data.table::fread("/well/mars/users/uvy786/dl_mouse_human/data/mouse_human/data.ign/MouseExpressionMatrix_voxel_coronal_maskcoronal_log2_grouped_imputed_labelled.csv", header=TRUE))
dfExprHuman <- as_tibble(data.table::fread("/well/mars/users/uvy786/dl_mouse_human/data/mouse_human/data.ign/HumanExpressionMatrix_samples_pipeline_abagen_labelled.csv"))

# Get gene names
genes <- colnames(dfExprMouse)[!str_detect(colnames(dfExprMouse), "Region")]

# Extract mouse voxel labels
dfLabelsMouse <- dfExprMouse %>%
  select(contains("Region")) %>%
  mutate(VoxelID = str_c("V", 1:nrow(.)))

# Extract human sample labels
dfLabelsHuman <- dfExprHuman %>%
  select(contains("Region")) %>%
  mutate(SampleID = str_c("S", 1:nrow(.)))

# Load mouse/human labels
load("/well/mars/users/uvy786/MouseHumanTranscriptomicSimilarity/data/TreeLabelsReordered.RData")

# Import MICe data tree
load("/well/mars/users/uvy786/MouseHumanTranscriptomicSimilarity/AMBA/data/MouseExpressionTree_DSURQE.RData")
treeMouse <- Clone(treeMouseExpr)



# Remove white matter and ventricles
pruneAnatTree(treeMouse, nodes = c("fiber tracts", "ventricular systems"), method = "AtNode")

# Import atlas, mask, and template images in AMBA space
dsurqeLabels_200um <- mincGetVolume("/well/mars/users/uvy786/MouseHumanTranscriptomicSimilarity/AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc")
dsurqeMask_200um <- mincGetVolume("/well/mars/users/uvy786/MouseHumanTranscriptomicSimilarity/AMBA/data/imaging/coronal_200um_coverage_bin0.8.mnc")
dsurqeAnat_200um <- mincGetVolume("/well/mars/users/uvy786/MouseHumanTranscriptomicSimilarity/AMBA/data/imaging/DSURQE_CCFv3_average_200um.mnc")

# Get mask for grey matter 
indMask <- dsurqeMask_200um == 1
indGM <- dsurqeLabels_200um %in% treeMouse$`Basic cell groups and regions`$label
indVoxels <- indMask & indGM

# Create an empty array and assign IDs to non zero voxels
emptyArray <- numeric(length(dsurqeLabels_200um))
names(emptyArray) <- character(length(emptyArray))
voxelsNames <- str_c("V", 1:sum(indVoxels))
names(emptyArray)[indVoxels] <- voxelsNames


