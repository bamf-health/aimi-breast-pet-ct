import sys, os
import glob
import pandas as pd
from tqdm.auto import tqdm
import SimpleITK as sitk
import numpy as np
import cv2
from image_processing.register_resample import registration
from skimage import measure, filters
import logging
from pathlib import Path


class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)


class Breast:
    def __init__(self, context):
        self.context = context
        self.summary = []

    def _get_path(self, *paths):
        """
        Concatenate paths to create a complete file path.

        Args:
            *paths: Variable number of path segments.

        Returns:
            str: Complete file path.
        """
        return os.path.join(*paths)

    def get_ensemble(self, save_path, label, num_folds=5, th=0.6):
        """
        Perform ensemble segmentation on medical image data.

        Args:
            save_path (str): Path to the directory where segmentation results will be saved.
            label (int): Label value for the segmentation.
            num_folds (int, optional): Number of folds for ensemble. Default is 5.
            th (float, optional): Threshold value. Default is 0.6.

        Returns:
            np.ndarray: Segmentation results.
        """
        for fold in range(num_folds):
            output_file = f"{self.context.pred_name}_{fold}.nii.gz"
            ip_path = os.path.join(save_path, output_file)
            seg_data = sitk.GetArrayFromImage(sitk.ReadImage(ip_path))
            seg_data[seg_data != label] = 0
            seg_data[seg_data == label] = 1
            if fold == 0:
                segs = np.zeros(seg_data.shape)
            segs += seg_data
        segs = segs / 5
        segs[segs < th] = 0
        segs[segs >= th] = 1
        return segs

    def mask_labels(self, labels, ts):
        """
        Create a mask based on given labels.

        Args:
            labels (list): List of labels to be masked.
            ts (np.ndarray): Image data.

        Returns:
            np.ndarray: Masked image data.
        """
        lung = np.zeros(ts.shape)
        for lbl in labels:
            lung[ts == lbl] = 1
        return lung

    def bbox2_3D(self, img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax

    def n_connected(self, img_data):
        """
        Get the largest connected component in a binary image.

        Args:
            img_data (np.ndarray): image data.

        Returns:
            np.ndarray: Processed image with the largest connected component.
        """
        img_filtered = np.zeros(img_data.shape)
        blobs_labels = measure.label(img_data, background=0)
        lbl, counts = np.unique(blobs_labels, return_counts=True)
        lbl_dict = {}
        for i, j in zip(lbl, counts):
            lbl_dict[i] = j
        sorted_dict = dict(sorted(lbl_dict.items(), key=lambda x: x[1], reverse=True))
        count = 0

        for key, value in sorted_dict.items():
            if count >= 1 and count <= 2 and value > 20:
                print(key, value)
                img_filtered[blobs_labels == key] = 1
            count += 1

        img_data[img_filtered != 1] = 0
        return img_data

    def arr_2_sitk_img(self, arr, ref):
        """
        Convert numpy array to SimpleITK image.

        Args:
            arr (np.ndarray): Input image data as a numpy array.
            ref: Reference image for copying information.

        Returns:
            sitk.Image: Converted SimpleITK image.
        """
        op_img = sitk.GetImageFromArray(arr)
        op_img.CopyInformation(ref)

        return op_img

    def infer_5_folds(self, save_path, ct_path, pt_path):
        """
        Perform inference for 5 folds using nnUNet.

        Args:
            save_path (str): Path to save inference results.
            ct_path (str): Path to CT image.
            pt_path (str): Path to PT image.
        """
        for fold in range(5):
            context = {
                'checkpoint_path': base_models_path,
                'input_file': input_ct_file,
                'pt_file': None,
                'prediction_save': save_path,
                'predict_aug': False,
                'softmax': False,
                'organ_name': f'{series_id}_{folds}',
                'fold': folds,
            }

            output_file = f"{self.context.pred_name}_{fold}"
            logging.info(f"inferring for fold {fold}")
            if not os.path.isfile(self._get_path(save_path, f"{output_file}.nii.gz")):
                os.system(
                    f"python /home/gmurugesan/projects/nnunet_segmentation_inference/nnunet_infer.py {self.context.checkpoint_path} --input_file {ct_path} --pt_file {pt_path} --prediction_save {save_path} --organ_name {output_file} --fold {fold}"
                )

    def postprocessing(self, save_path, ct_path):
        """
        Perform postprocessing and writes simpleITK Image

        Args:
            save_path (str): Path to save inference results.
            ct_path (str): Path to CT image.
        Returns:
            None
        """
        post_img = self._get_path(save_path, f"{self.context.ensemble_name}.nii.gz")
        if not os.path.isfile(post_img):
            ts_data = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(save_path, "totalsegmentator.nii.gz"))
            )
            ts_abdominal = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(save_path, "totalsegmentator.nii.gz"))
            )
            ts_data[ts_data > 1] = 1
            lesions = self.get_ensemble(save_path, {self.context.pred_name}, 9, th=0.6)
            op_data = np.zeros(ts_data.shape)
            ref = sitk.ReadImage(ct_path)
            ct_data = sitk.GetArrayFromImage(ref)
            op_data[lesions == 1] = 1
            th = np.min(ct_data)
            op_data[ct_data == th] = 0  # removing predicitons where CT not available
            # Use the coordinates of the bounding box to crop the 3D numpy array.
            ts_abdominal[ts_abdominal > 4] = 0
            ts_abdominal[ts_abdominal > 1] = 1
            if ts_abdominal.max() > 0:
                x1, x2, y1, y2, z1, z2 = self.bbox2_3D(ts_abdominal)
            # Create a structuring element with ones in the middle and zeros around it
            structuring_element = np.ones((3, 3))

            # Dilate the array with the structuring element
            op_temp = cv2.dilate(ts_data, structuring_element, iterations=5)
            op_temp = cv2.erode(op_temp, structuring_element, iterations=5)
            op_data[op_temp == 1] = 0
            if ts_abdominal.max() > 0:
                op_data[x1:x2, y1:, :] = 0
            op_data[0:3, :, :] = 0
            op_data = self.n_connected(op_data)
            op_img = self.arr_2_sitk_img(op_data, ref)
            sitk.WriteImage(op_img, post_img)

    def infer_totalsegmentator(
        self,
        save_path,
        ct_path,
    ):
        """
        Perform inference using TotalSegmentator on the provided CT image.

        Args:
            save_path (str): Path to save the segmentation result.
            ct_path (str): Path to the CT image for segmentation.
        """
        if not os.path.isfile(self._get_path(save_path, "totalsegmentator.nii.gz")):
            os.system(
                f"TotalSegmentator -i {ct_path} -o {self._get_path(save_path, 'totalsegmentator.nii.gz')} --ml"
            )

    def _append(self, ct_path, pt_path, save_path):
        """
        Append information about a segmentation result to the summary.

        Args:
            ct_path (str): Path to the CT image.
            pt_path (str): Path to the PT image.
            save_path (str): Path where the segmentation result is saved.
        """
        self.summary.append(
            {
                "CT": ct_path,
                "PT": pt_path,
                "Segmentation": self._get_path(
                    save_path, f"{self.context.ensemble_name}.nii.gz"
                ),
            }
        )

    def _registration(self, pt_path, ct_path):
        rct = str(Path(ct_path).parent / str("r" + Path(ct_path).stem + ".gz"))
        if not os.path.isfile(rct):
            registration(pt_path, ct_path)
        logging.info(rct)
        return rct

    def save_summary(self):
        """
        Save the inference summary to a CSV file.

        This method converts the collected inference summary into a pandas DataFrame
        and saves it as a CSV file in the specified directory with a filename
        containing the CT type.

        Note:
            The method must be called after completing the inference process.
        """
        inference_summary = pd.DataFrame(self.summary)
        inference_summary.to_csv(
            self._get_path(
                self.context.base_dir,
                f"inference_summary_ensemble_{self.context.CT_type}.csv",
            )
        )

    def curate_qin_breast(self, path):
        df = pd.read_pickle(path)
        df_pt_ct = df.loc[(df["Modality"] == "PT") | (df["Modality"] == "CT")]
        return df_pt_ct

    def get_inference(self):
        summary = []
        df_pt_ct = self.curate_qin_breast(self.context.scan_data)
        for idx in tqdm(range(0, 220, 2)):
            df_sub = df_pt_ct.iloc[idx : idx + 2]
            if df_sub.iloc[0]["StudyInstanceUID"] == df_sub.iloc[1]["StudyInstanceUID"]:
                ct_df = df_sub.loc[df_sub["Modality"] == "CT"]
                pt_df = df_sub.loc[df_sub["Modality"] == "PT"]
                ct_series = ct_df["SeriesInstanceUID"].values[0]
                pt_series = pt_df["SeriesInstanceUID"].values[0]
                pat_id = ct_df["PatientID"].values[0]
                studyid = ct_df["StudyInstanceUID"].values[0]
                pt_path = os.path.join(
                    self.context.base_dir, pat_id, studyid, f"{pt_series}.nii.gz"
                )
                ct_path = os.path.join(
                    self.context.base_dir, pat_id, studyid, f"{ct_series}.nii.gz"
                )
                ct_path = self._registration(pt_path, ct_path)
                ct_path = os.path.join(
                    self.context.base_dir, pat_id, studyid, f"r{ct_series}.nii.gz"
                )
                save_path = os.path.join(self.context.base_dir, pat_id, studyid)
                self.infer_5_folds(save_path, ct_path, pt_path)
                self.infer_totalsegmentator(save_path, ct_path)
                self.postprocessing(save_path, ct_path)
                self._append(
                    ct_path,
                    pt_path,
                    os.path.join(save_path, f"{self.context.ensemble_name}.nii.gz"),
                )
        self.save_summary()
        return summary
