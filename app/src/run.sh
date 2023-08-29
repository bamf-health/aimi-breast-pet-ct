#!/usr/bin/env bash
# arg1: dir with dicom files for single ct series
# arg2: output dir

mkdir /tmp/nii-input
python3 /app/convert-dcm-to-nii.py $1 /tmp/nii-input/ct_0000.nii.gz

# nnunet predict all folds seperatly
nnUNet_predict -i /tmp/nii-input -o /tmp/folds -t Task779_Kidneys_KIRC -m 3d_fullres -f 0
mv /tmp/folds/ct.nii.gz /tmp/folds/ct_fold0.nii.gz
nnUNet_predict -i /tmp/nii-input -o /tmp/folds -t Task779_Kidneys_KIRC -m 3d_fullres -f 1
mv /tmp/folds/ct.nii.gz /tmp/folds/ct_fold1.nii.gz
nnUNet_predict -i /tmp/nii-input -o /tmp/folds -t Task779_Kidneys_KIRC -m 3d_fullres -f 2
mv /tmp/folds/ct.nii.gz /tmp/folds/ct_fold2.nii.gz
nnUNet_predict -i /tmp/nii-input -o /tmp/folds -t Task779_Kidneys_KIRC -m 3d_fullres -f 3
mv /tmp/folds/ct.nii.gz /tmp/folds/ct_fold3.nii.gz
nnUNet_predict -i /tmp/nii-input -o /tmp/folds -t Task779_Kidneys_KIRC -m 3d_fullres -f 4
mv /tmp/folds/ct.nii.gz /tmp/folds/ct_fold4.nii.gz

# ensemble and post process
python /app/ensemble.py /tmp/folds /tmp/label.nii.gz

# copy output to output dir
python /app/copy-to-series-dir.py /tmp/label.nii.gz $1 $2