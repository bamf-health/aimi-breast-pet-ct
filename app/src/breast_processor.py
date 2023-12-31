import os
import SimpleITK as sitk
import numpy as np
import cv2
from skimage import measure


class BreastPostProcessor:
    def __init__(self):
        pass

    def get_ensemble(self, save_path, organ_name_prefix, label, num_folds=5, th=0.6):
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
            inp_seg_file = f"{organ_name_prefix}_{fold}.nii.gz"
            inp_seg_path = os.path.join(save_path, inp_seg_file)
            seg_data = sitk.GetArrayFromImage(sitk.ReadImage(inp_seg_path))
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

    def postprocessing(self, save_path, ct_path, total_seg_path, out_file_path, organ_name_prefix, breast_label):
        """
        Perform postprocessing and writes simpleITK Image

        Args:
            save_path (str): Path to save inference results.
            ct_path (str): Path to CT image.
            total_seg_path (str): Path to segmentation output from total-segmentator
            out_file_path (str): Path to save output to
            organ_name_prefix (str): base name of the output mask from nnUNet model. This is used to fetch output masks for each fold
            breast_label (str): label of breast assigned in AIMI dataset

        Returns:
            None
        """
        if not os.path.isfile(out_file_path):
            ts_data = sitk.GetArrayFromImage(sitk.ReadImage(total_seg_path))
            ts_abdominal = sitk.GetArrayFromImage(sitk.ReadImage(total_seg_path))
            ts_data[ts_data > 1] = 1
            lesions = self.get_ensemble(save_path, organ_name_prefix, label=int(breast_label), th=0.6)
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
            sitk.WriteImage(op_img, out_file_path)
