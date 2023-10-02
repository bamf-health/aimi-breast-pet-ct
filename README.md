## Summary

Resources to create a container that runs nnUNet model inference on AIMI collection out of the box

### Running instructions
* Create an `input_dir_ct` containing list of `dcm` files corresponding to a given series_id for CT modality
* Create an `input_dir_pt` containing list of `dcm` files corresponding to a given series_id for PT modality 
* Create an `output_dir` to store the output from the model. This is a shared directory mounted on container at run-time. Please assign write permissions to this dir for container to be able to write data
* Next, pull the image from dockerhub:
  * `docker pull bamfhealth/bamf_nnunet_pt_ct_breast:latest`

* Finally, let's run the container:
  * `docker run --gpus all -v {input_dir_ct}:/app/data/input_data/ct:ro -v {input_dir_pt}:/app/data/input_data/pt -v {output_dir}:/app/data/output_data bamfhealth/bamf_nnunet_pt_ct_breast:latest`
* Once the job is finished, the output inference mask(s) would be available in the `{output_dir}` folder
* Expected output from after successful container run is:
  * `{output_dir}/seg_ensemble_primary.dcm` -- if nifti to dcm conversion is a success
  * `{output_dir}/seg_lesions_ensemble.nii.gz` -- if nifti to dcm conversion is a failure

# Breast PET/CT Segmentation

We used our FDG-PET/CT lesion segmentation model trained on [AutoPET 2023](https://autopet.grand-challenge.org/) dataset to segment lesions for each of the five folds and ensemble the lesion segments. We then used segmentations from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) to limit the predicted lesions to only those in the breast area.

The [model_performance](model_performance.ipynb) notebook contains the code to evaluate the model performance on [QIN-Breast](https://wiki.cancerimagingarchive.net/display/Public/QIN-Breast) collection against a validation evaluated by a radiologist and a non-expert.

## Running the model

#TODO

### Build container from pretrained weights

#TODO

### Running inference

By default the container takes an input directory that contains DICOM files of PET/CT scans from the [QIN-Breast](https://wiki.cancerimagingarchive.net/display/Public/QIN-Breast) collections, and an output directory where DICOM-SEG files will be placed. To run on multiple scans, place DICOM files for each scan in a separate folder within the input directory. The output directory will have a folder for each input scan, with the DICOM-SEG file inside.

example:

#TODO

There is an optional `--nifti` flag that will take nifti files as input and output.

#### Run inference on IDC Collections

This model was run on PET/CT scans from the [QIN-Breast](https://wiki.cancerimagingarchive.net/display/Public/QIN-Breast) collection. The AI segmentations and corrections by a radioloist for 10% of the dataset are available in the breast-fdg-pet-ct.zip file on the [zenodo record](https://zenodo.org/record/8352041)

You can reproduce the results with the [run_on_idc_data](run_on_idc_data.ipynb) notebook on google colab.

### Training your own weights

Refer to the [training instructions](training.md) for more details. #TODO
