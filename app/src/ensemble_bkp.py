#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import numpy as np
import SimpleITK as sitk
from skimage import measure


def perform_ensemble(folds: List[Path], label: int):
    """
    Create ensemble based on majority vote
    :param: folds -- list of nii.gz files, each being an output of the nnUNet model fold
    :param: label -- label of the target segment to perform ensemble for
    returns: aggregated segmented mask
    """
    segs: np.ndarray = None
    for fold in folds:
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(str(fold)))
        seg_data[seg_data != int(label)] = 0
        seg_data[seg_data == int(label)] = 1
        if segs is None:
            segs = np.zeros(seg_data.shape)
        segs += seg_data
    segs = segs / len(folds)
    segs[segs < 0.6] = 0
    segs[segs >= 0.6] = 1
    return segs


def n_connected(img_data):
    """
    Keep largest connected component of the segmentation and remove all other smaller components
    :param: img_data: np.NdArray on which connected component analysis to be applied
    returns: <np.NdArray>
    """

    img_data_mask = np.zeros(img_data.shape)
    img_data_mask[img_data > 0] = 1
    img_filtered = np.zeros(img_data_mask.shape)
    blobs_labels = measure.label(img_data_mask, background=0)
    lbl, counts = np.unique(blobs_labels, return_counts=True)
    lbl_dict = {}
    for i, j in zip(lbl, counts):
        lbl_dict[i] = j
    sorted_dict = dict(sorted(lbl_dict.items(), key=lambda x: x[1], reverse=True))
    count = 0

    for key, value in sorted_dict.items():
        if count >= 1 and count <= 2:
            print(key, value)
            img_filtered[blobs_labels == key] = 1
        count += 1

    img_data[img_filtered != 1] = 0
    return img_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folds_dir",
        type=Path,
        help="input directory containing folds of predictions. expect 5 folds, each fold is a *.nii.gz file",
    )
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    folds = list(args.folds_dir.glob("*.nii.gz"))
    assert len(folds) == 5

    kidney = get_ensemble(folds, 1)
    tumor = get_ensemble(folds, 2)
    cysts = get_ensemble(folds, 3)
    ref = sitk.ReadImage(str(folds[0]))
    op_data = np.zeros(kidney.shape)
    op_data[kidney == 1] = 1
    op_data[tumor == 1] = 2
    op_data[cysts == 1] = 3
    op_data = n_connected(op_data)
    op_img = sitk.GetImageFromArray(op_data)
    op_img.CopyInformation(ref)
    sitk.WriteImage(op_img, str(args.output_file))