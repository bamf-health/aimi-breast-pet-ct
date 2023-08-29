#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import pydicom
import SimpleITK as sitk


class DicomToNiiConverter:
    def __init__(self) -> None:
        pass

    def dcm_to_niix(self, dcm_dir: Path, nii_path: Path):
        """uses dcm2niix to convert a series of dicom files to a nifti file"""
        dcm_dir = Path(dcm_dir)
        nii_path = Path(nii_path)
        with TemporaryDirectory() as tmpdir:
            args = [
                "dcm2niix",
                "-o",
                tmpdir,
                "-z",
                "y",
                str(dcm_dir.resolve()),
            ]
            subprocess.run(args, check=True)

            nii_files = list(Path(tmpdir).glob("*Eq_*.nii.gz"))
            if len(nii_files) > 1:
                raise ValueError(f"Expected 1 Eq_*.nii.gz file, found {len(nii_files)}")
            elif len(nii_files) == 1:
                shutil.move(nii_files[0], nii_path)
                return
            # no Eq images
            nii_files = list(Path(tmpdir).glob("*.nii.gz"))
            if len(nii_files) > 1:
                raise ValueError(f"Expected 1 *.nii.gz file, found {len(nii_files)}")
            elif len(nii_files) == 1:
                shutil.move(nii_files[0], nii_path)
                return
            raise ValueError(f"Expected 1 *.nii.gz file, found 0")


    def dcm_to_nii(self, dcm_dir: Path, nii_path: Path) -> bool:
        """uses SimpleITK to convert a series of dicom files to a nifti file"""
        # sort the files to hopefully make conversion a bit more reliable
        files = []
        dcm_dir = Path(dcm_dir)
        nii_path = Path(nii_path)
        for f in dcm_dir.glob("*.dcm"):
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            slicer_loc = ds.SliceLocation if hasattr(ds, "SlicerLocation") else 0
            files.append((slicer_loc, f))
        slices = sorted(files, key=lambda s: s[0])
        ordered_files = [x[1] for x in slices]

        with TemporaryDirectory() as tmp_dir:
            ptmp_dir = Path(tmp_dir)
            for i, f in enumerate(ordered_files):
                shutil.copy(f, ptmp_dir / f"{i}.dcm")
            try:
                # load in with SimpleITK
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(tmp_dir)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()

                nii_path.parent.mkdir(parents=True, exist_ok=True)
                # save as nifti
                sitk.WriteImage(
                    image, str(nii_path.resolve()), useCompression=True, compressionLevel=9
                )
            except:
                return False
            return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert dicom files to nifti file. Supports single series"
    )
    parser.add_argument(
        "dcm_dir", type=Path, help="input directory containing dicom files"
    )
    parser.add_argument("nii_path", type=Path, help="output directory for nifti files")
    parser.add_argument(
        "--niix",
        action="store_true",
        help="use dcm2niix instead of SimpleITK for conversion",
    )
    args = parser.parse_args()

    converter = DicomToNiiConverter()
    if args.niix:
        converter.dcm_to_niix(args.dcm_dir, args.nii_path)
    else:
        converter.dcm_to_nii*(args.dcm_dir, args.nii_path)
