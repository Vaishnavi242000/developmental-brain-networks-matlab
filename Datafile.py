"""
Datafile.py

Downloads the data for the developmental connectivity analysis and prepares
everything run_connectivity.m needs, saved into ./data:

    manifest.csv            one row per participant: ID, age, group, and paths
                            to the preprocessed fMRI and confound files
    atlas_schaefer100.nii   Schaefer 100-parcel atlas on the fMRI voxel grid
    atlas_labels.csv        parcel names, which encode the Yeo network

Dataset: Richardson et al. (2018), movie-watching fMRI in children and adults
(OpenNeuro ds000228), preprocessed with fMRIPrep and distributed via nilearn.

Usage:
    pip install nilearn pandas
    python Datafile.py
"""

from pathlib import Path

import pandas as pd
from nilearn import datasets, image

# Number of participants to download. nilearn keeps the same child/adult ratio
# as the full dataset, so 40 gives roughly 31 children and 9 adults.
# Set to 155 to use everyone.
N_SUBJECTS = 40

OUT = Path("data")
OUT.mkdir(exist_ok=True)


def find_col(df, name):
    """Look up a column by name, ignoring capitalization.

    The phenotype column names aren't consistent across nilearn versions
    (e.g. "Age" vs "age"), so this avoids hard-coding the exact spelling.
    """
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)}")


# Step 1: fMRI data and participant information

dev = datasets.fetch_development_fmri(n_subjects=N_SUBJECTS)
pheno = pd.DataFrame(dev.phenotypic).reset_index(drop=True)

# Absolute paths let MATLAB find the files no matter which folder it runs from.
manifest = pd.DataFrame({
    "participant_id": pheno[find_col(pheno, "participant_id")],
    "age":            pheno[find_col(pheno, "Age")],
    "group":          pheno[find_col(pheno, "Child_Adult")].astype(str).str.lower(),
    "func":           [str(Path(f).resolve()) for f in dev.func],
    "confounds":      [str(Path(c).resolve()) for c in dev.confounds],
})
manifest.to_csv(OUT / "manifest.csv", index=False)


# Step 2: Brain atlas

# The atlas ships at 2 mm resolution, but the fMRI data are lower resolution.
# Resampling onto the fMRI grid means each voxel gets one parcel label.
# Nearest-neighbor interpolation keeps the labels as whole numbers instead of
# blending neighboring parcels together.
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7,
                                              resolution_mm=2)
atlas = image.resample_to_img(schaefer.maps, dev.func[0],
                              interpolation="nearest")
atlas.to_filename(OUT / "atlas_schaefer100.nii")

# Older nilearn versions return labels as bytes, and newer ones include a
# "Background" entry; normalize both so the list has exactly one name per parcel.
labels = [l.decode() if isinstance(l, bytes) else str(l) for l in schaefer.labels]
labels = [l for l in labels if l.lower() != "background"]

pd.DataFrame({"index": range(1, len(labels) + 1), "label": labels}) \
  .to_csv(OUT / "atlas_labels.csv", index=False)

# Summary

n_children = (manifest.group == "child").sum()
n_adults = (manifest.group == "adult").sum()
print(f"Saved {len(manifest)} subjects ({n_children} children, {n_adults} adults) "
      f"and a {len(labels)}-parcel atlas to ./data")