import yaml
import argparse
import os
from pathlib import Path
from converter_utils import DicomToNiiConverter
from ensemble import perform_ensemble, n_connected
from registration_utils import registration
from bamf_nnunet_inference import BAMFnnUNetInference


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


def post_process(folds_dir, target_path, label, num_folds):
    folds = list(folds_dir.glob("*.nii.gz"))
    assert len(folds) == num_folds
    breast_label = perform_ensemble(folds, label)


def run_nnunet(input_data, output_data, num_folds=5):
    """
    Convert list of dcm files to a single nii.gz file
    :param: input_data - dir containing list of dcm files
    :param: output_data - dir to write segmented dcm masks too
    :param: weights_dir - dir containing nnUNet model weights
    """    
    temp_nii_dir = "/tmp/nii-input"
    Path(temp_nii_dir).mkdir(parents=True, exist_ok=True)
    temp_nii_path = os.path.join(temp_nii_dir, "ct_0000.nii.gz")

    temp_folds_dir = "/tmp/folds"
    Path(temp_folds_dir).mkdir(parents=True, exist_ok=True)
    temp_label_path = os.path.join(temp_folds_dir, "label.nii.gz")

    # Load env vars (model weights and task name)
    WEIGHTS_FOLDER = os.environ["WEIGHTS_FOLDER"]
    TASK_NAME = os.environ["TASK_NAME"]

    # convert dcm to nii
    converter = DicomToNiiConverter()
    converter.dcm_to_nii(input_data, temp_nii_path)

    # Prepare config for nnUNet model
    BREAST_LABEL = "1"
    base_models_path = f"/mnt/nfs/ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_trained_models/nnUNet/3d_fullres/{TASK_ID}/nnUNetTrainerV2__nnUNetPlansv2.1"
    liver_inference = BAMF_organseg()


    # nnunet predict all folds seperatly
    for fold_idx in range(num_folds):
        infer_fold = f"nnUNet_predict -i {temp_nii_dir} -o {temp_folds_dir} -t {TASK_NAME} -m 3d_fullres -f {fold_idx}"
        move_data = f"mv {temp_folds_dir}/ct.nii.gz {temp_folds_dir}/ct_fold{fold_idx}.nii.gz"
        os.system(infer_fold)
        os.system(move_data)

    # ensemble and post process
    post_process(folds_dir=temp_folds_dir, target_path=temp_label_path, num_folds=num_folds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display YAML configuration")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)

    # Load arguments from config
    general = config.get("general", {})
    data_base_dir = general.get("data_base_dir")
    modules = config.get("modules", {})
    nnunet_runner = modules.get("NNUnetRunner", {})
    source_dir = nnunet_runner.get("source_dir")
    target_dir = nnunet_runner.get("target_dir")
    num_folds = int(nnunet_runner.get("num_folds"))
    source_dir = os.path.join(data_base_dir, source_dir)
    target_dir = os.path.join(data_base_dir, target_dir)

    # Run the model
    run_nnunet(source_dir, target_dir, num_folds)





