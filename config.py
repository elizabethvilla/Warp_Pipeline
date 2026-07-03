# ==================================================
# ============ USER-EDITABLE SETTINGS ==============
# ==================================================

# [SNAPSHOT CONFIG] 
# This file is specific to this dataset directory.
# Modifications here ONLY affect the current dataset run.

# --- General Settings ---
dataset_name = "260626_iN_HT_screen"
raw_directory = "/data/Microscopy/Titan/Siyu" 
# Path to where you would like to save your raw data
# Titan2 data will be moved there and there might be no copy of your raw data in the original place
frame_folder = "frames"
mdoc_folder = "mdocs"
gain_ref = "None"   # if not .gain
tomo_match_string = "Position" 

# --- Key Acquisition Parameters ---
angpix = 1.561
dose = 4.24
tilt_axis_angle = 84.48
thickness_pxl = 3000
camera_type = "Falcon4" # Switch between "K3" or "Falcon4"

# --- Falcon4 Specific Settings ---
# The source directory containing raw .eer and .eer.mdoc files
falcon4_source_dir = "/data/Microscopy/titan2/Villa_20260625_110712"
falcon4_eer_ngroups = 16

# --- K3 Specific Settings ---
k3_frame_num = 8

# ==================================================
# ============== ADVANCED PARAMETERS  ==============
# ==================================================

# --- Computing Resources ---
import os
gpu_devices = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
jobs_per_gpu = 4
etomo_cpu_cores = 8
gapstop_workers = 24

# --- Base Path ---
base_dir = "/data/workspace/Siyu/Titan1_Processing"

# --- Warp Pipeline Parameters ---
pipeline_params = {}
extra_create_args = []

if camera_type == "K3":
    pipeline_params["extension"] = "*.tif"
    pipeline_params["m_grid_frames"] = k3_frame_num
    pipeline_params["original_x_y_size"] = (5760, 4092)
    extra_create_args.append("--gain_flip_x")

elif camera_type == "Falcon4":
    pipeline_params["extension"] = "*.eer"
    pipeline_params["m_grid_frames"] = falcon4_eer_ngroups
    pipeline_params["original_x_y_size"] = (4096, 4096)
    extra_create_args.extend(["--eer_ngroups", str(falcon4_eer_ngroups)])

else:
    raise ValueError(f"Unsupported camera_type in config: {camera_type}")

pipeline_params["extra_create_args"] = extra_create_args

# --- eTomo Settings ---
FINAL_NEWSTACK_BIN = 8
final_x_size = pipeline_params["original_x_y_size"][0] // FINAL_NEWSTACK_BIN
final_y_size = pipeline_params["original_x_y_size"][1] // FINAL_NEWSTACK_BIN

etomo_params = {
    "setupset.copyarg.pixel": angpix / 10.0,
    "setupset.copyarg.rotation": tilt_axis_angle,
    # "comparam.newst.newstack.SizeToOutputInXandY": f"{final_x_size},{final_y_size}",
}

use_dynamic_patch_size = True
patch_size_division_factor = 4
possible_patch_sizes = [256, 512, 1024, 2048]
default_patch_size = 512

# --- ISONet Settings ---
isonet_params = {
    "cube_size": 64,
    "crop_size": 96,
    "number_subtomos": 8,
    "iterations": 30,
    "noise_level": [0.1, 0.15, 0.2, 0.25],
    "noise_start_iter": [10, 15, 20, 25],
    "density_percentage": 40,
    "std_percentage": 40,
    "z_crop": 0.14,
    "batch_size": len(gpu_devices)
}

# --- ISONet2 Settings ---
isonet2_params = {
    "cube_size": 96,
    "density_percentage": 50,
    "std_percentage": 80,
    "z_crop": 0.2,
    "epochs": 100,
    "mw_weight": 200,
    "batch_size": len(gpu_devices),
    "tilt_min": -54,   # set per dataset
    "tilt_max": 54,    # set per dataset
}

# --- Cryolo Settings ---
cryolo_params = {
    "prep": {
        "enable": True,
        "star_file": f"{base_dir}/251113_HSC_rest/relion32_7p48/Refine3D/ms1c246_mr1/run_data.star",
        "bin_factor": 1
    },
    "threshold": 0.25,
    "min_connections": 5,
    "batch_size": len(gpu_devices)*2
}

# --- 3D template matching Settings ---
template_matching_params = {
    "tomo_angpix": angpix * FINAL_NEWSTACK_BIN,
    # "tomo_angpix": 10,
    "subdivisions": 3,
    "template_path": f"{base_dir}/260401_Janelia_HSC/gapstop/round_1/ini.mrc",
    "template_diameter": 350,
    "peak_distance": 175,
    "symmetry": "C1",
    "input_data": "NONE",  # Path to the matching.txt file; if "NONE", will run with full tomoset
    "reuse_results": True
}

# --- gapstop Settings ---
gapstop_params = {
    "template": "ini_inv.mrc",
    "mask": "spherical_rad26_gaus2.mrc",
    "angle_file": "angles_10_C1.txt",
    "particle_diameter": 28,
    "scores_threshold": 0.07,
    "sigma_threshold": None,
    "cluster_size": None,
    "n_particles": None,
    "symmetry": "c1",
}

# --- particle export Settings ---
subtomo_params = {
    "3d": True,
    "--input_directory": "warp_tiltseries/matching/filtered",
    "--input_pattern": "*.star",
    "--coords_angpix": angpix * FINAL_NEWSTACK_BIN,
    "--output_star": "relion32_bin4/bin4.star",
    "--output_angpix": angpix * 4,
    "--output_processing": "relion32_bin4",
    "--box": 84,
    "--diameter": 350
}

# --- m refine Settings ---
m_refine_params = {
    "directory": "3p74_to1p87_3set_try2",
    "population_name" : "3set",
    "relion_folder" : f"{base_dir}/ribo_relion/relion5_3p74A",
    "input_angpix" : 3.74,
    "source_names" : [
        {"dataset": "251028_HSC_2d", "name":"m_full_251028"},
        {"dataset": "251113_HSC_rest", "name":"m_full_251113"},
        {"dataset": "20250820_HSC_4hr", "name":"m_full_20250820"},
    ],
    "species": [
        {"name":"ribosome", "job":"mr2_global","mask":"ini_2_3_6"},
        # {"name":"ribosome_eEF2","job":"eEF2_c25_mr1","mask":"eEF2_c2_3_3_6"},
        # {"name": "ribosome_AT", "job": "AT_c3_mr1", "mask": "AT_c3_3_3_6"},
        # {"name": "ribosome_AA", "job": "AA_c4_mr1", "mask": "AA_c4_3_3_6"},
        # {"name": "ribosome_P", "job": "P_c6_mr1", "mask": "P_c6_3_3_6"} 
    ],
}