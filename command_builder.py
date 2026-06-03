#!/usr/bin/env python3

from pathlib import Path

import os
import shlex
import sys
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
import config as cfg

def build_reconstruction_command(jobs_per_gpu, gpu_devices):
    """Builds the command for the reconstruction stage."""
    cmd = [
        "WarpTools", "ts_reconstruct",
        "--settings", "warp_tiltseries.settings",
        "--angpix", str(cfg.angpix * cfg.FINAL_NEWSTACK_BIN),
        "--halfmap_frames",
        "--perdevice", str(jobs_per_gpu)
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    return cmd

def build_isonet_commands(isonet_params, gpu_devices, tomo_folder="tomoset"):
    """Builds the list of commands for the ISONet stage."""
    pixel_size = cfg.angpix * cfg.FINAL_NEWSTACK_BIN
    gpu_ids = ','.join(str(d) for d in gpu_devices)
    noise_level = ','.join(str(x) for x in isonet_params['noise_level'])
    noise_start_iter = ','.join(str(x) for x in isonet_params['noise_start_iter'])
    
    commands = [
        f"isonet.py prepare_star {tomo_folder} --output_star tomogram.star --pixel_size {pixel_size} --number_subtomos {isonet_params['number_subtomos']}",
        f"isonet.py make_mask tomogram.star --mask_folder mask --density_percentage {isonet_params['density_percentage']} --std_percentage {isonet_params['std_percentage']} --z_crop {isonet_params['z_crop']}",
        f"isonet.py extract tomogram.star --cube_size {isonet_params['cube_size']}",
        f"isonet.py refine subtomo.star --gpuID {gpu_ids} --iterations {isonet_params['iterations']} --noise_level {noise_level} --noise_start_iter {noise_start_iter} --log_level info --batch_size {isonet_params['batch_size']}",
        f"isonet.py predict tomogram.star ./results/model_iter{isonet_params['iterations']}.h5 --gpuID {gpu_ids} --cube_size {isonet_params['cube_size']} --crop_size {isonet_params['crop_size']} --log_level info --batch_size {isonet_params['batch_size']}",
    ]
    return commands

def build_isonet2_commands(isonet2_params, gpu_devices, defocus_list):
    """Builds the list of commands for the ISONet2 stage."""
    pixel_size = cfg.angpix * cfg.FINAL_NEWSTACK_BIN
    gpu_ids = ','.join(str(d) for d in gpu_devices)
    cube_size = isonet2_params['cube_size']
    defocus_str = "[" + ",".join(str(d) for d in defocus_list) + "]"

    commands = [
        f'isonet.py prepare_star --even even --odd odd --pixel_size {pixel_size} --tilt_min {isonet2_params["tilt_min"]} --tilt_max {isonet2_params["tilt_max"]} --defocus "{defocus_str}"',
        f"isonet.py denoise tomograms.star --CTF_mode network --gpuID {gpu_ids} --epochs {isonet2_params['epochs']}",
        f"isonet.py predict tomograms.star denoise/network_n2n_unet-medium_{cube_size}_full.pt --gpuID {gpu_ids}",
        f"isonet.py make_mask tomograms.star --density_percentage {isonet2_params['density_percentage']} --std_percentage {isonet2_params['std_percentage']} --z_crop {isonet2_params['z_crop']} --input_column rlnDenoisedTomoName",
        f"isonet.py refine tomograms.star --method isonet2-n2n --cube_size {cube_size} --epochs {isonet2_params['epochs']} --mw_weight {isonet2_params['mw_weight']} --CTF_mode network --bfactor 0 --gpuID {gpu_ids} --batch_size {isonet2_params['batch_size']}",
        f"isonet.py predict tomograms.star isonet_maps/network_isonet2-n2n_unet-medium_{cube_size}_full.pt --gpuID {gpu_ids}",
    ]
    return commands

def build_cryolo_commands(cryolo_params, gpu_devices):
    """Builds the list of commands for the Cryolo stage."""
    threshold = cryolo_params['threshold']
    min_connections = cryolo_params['min_connections']
    batch_size = cryolo_params['batch_size']
    gpu_ids = ' '.join(str(d) for d in gpu_devices)
    cpu = '32'
    cryolo_ad = "/software/repo/rhel9/cryolo/1.9.4/bin"
    cmd_ini = f"'{cryolo_ad}/python3.8' -u '{cryolo_ad}/cryolo_gui.py' --ignore-gooey"
    output_dir = Path(f"expand10_{threshold}_{min_connections}")

    commands = [
        f"{cmd_ini} config --train_image_folder '3DTM_pre/tomograms' --train_annot_folder '3DTM_pre/CBOX' --saved_weights_name 'cryolo_model_fromRelion_expand10.h5' -a 'PhosaurusNet' --input_size 1024 -nm 'STANDARD' --num_patches '1' --overlap_patches '200' --filtered_output 'filtered_tmp/' -f 'LOWPASS' --low_pass_cutoff '0.1' --janni_overlap '24' --janni_batches '3' --train_times '10' --batch_size '{batch_size}' --learning_rate '0.0001' --nb_epoch '200' --object_scale '5.0' --no_object_scale '1.0' --coord_scale '1.0' --class_scale '1.0' --debug --log_path 'logs/' -- 'config_cryolo.json' '64'",
        f"{cmd_ini} train -c 'config_cryolo.json' -w '5' -g {gpu_ids} -nc {cpu} --gpu_fraction '1.0' -e '10' -lft '2' --seed '10'",
        f"{cmd_ini} predict -c 'config_cryolo.json' -w 'cryolo_model_fromRelion_expand10.h5' -i tomograms -o '{output_dir}' -t '{threshold}' -g {gpu_ids} -d '0' -pbs '3' --gpu_fraction '1.0' -nc {cpu} --norm_margin '0.0' -sm 'LINE_STRAIGHTNESS' -st '0.95' -sr '1.41' -ad '10' --directional_method 'PREDICTED' -mw '100' --tomogram -tsr '-1' -tmem '0' -mn3d '2' -tmin '{min_connections}' -twin '-1' -tedge '0.4' -tmerge '0.8'"
    ]
    return commands, output_dir

def build_template_match_command(params, jobs_per_gpu, gpu_devices):
    """Builds the command for the 3D template matching stage."""
    cmd = [
        "WarpTools", "ts_template_match",
        "--settings", "warp_tiltseries.settings",
        "--tomo_angpix", str(params['tomo_angpix']),
        "--subdivisions", str(params['subdivisions']),
        "--template_path", str(params['template_path']),
        "--template_diameter", str(params['template_diameter']),
        "--peak_distance", str(params['peak_distance']),
        "--symmetry", str(params['symmetry']),
        "--perdevice", str(jobs_per_gpu),
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    
    list_file = Path(params['input_data'])
    if list_file.exists():
        cmd.extend(["--input_data", str(list_file)])
    
    if params.get('reuse_results', True):
        cmd.append("--reuse_results")
    
    return cmd

def build_gapstop_wedge_command(params, tomo_name, tomo_id):
    """Generates the command to create a wedge file using cryocat wedgeutils."""
    x = cfg.pipeline_params['original_x_y_size'][0]
    y = cfg.pipeline_params['original_x_y_size'][1]
    z = cfg.thickness_pxl
    pixel_size = cfg.angpix
    xml = f"{cfg.base_dir}/{cfg.dataset_name}/warp_tiltseries/{tomo_name}.xml"
    script_content = f"""from cryocat import wedgeutils

wedgeutils.create_wedge_list_sg(
    tomo_id = {tomo_id},
    tomo_dim = [{x}, {y}, {z}],
    pixel_size = {pixel_size},
    tlt_file = \"{xml}\",
    z_shift=0.0,
    ctf_file = \"{xml}\",
    ctf_file_type=\'warp\',
    dose_file = \"{xml}\",
    voltage=300.0,
    amp_contrast=0.07,
    cs=2.7,
    output_file = \"wedge_{tomo_name}.star\",
    drop_nan_columns=True
)
"""
    
    cmd = f"python3 -c {shlex.quote(script_content)}"
    return cmd

def build_gapstop_result_command(params, tomo_id, output_star):
    """Generates the command to extract particles from GapStop template matching results."""
    scores_path = f"tm_outputs/scores_{tomo_id-1}_{tomo_id}.mrc"
    angles_path = f"tm_outputs/angles_{tomo_id-1}_{tomo_id}.mrc"
    pixel_size = cfg.angpix * cfg.FINAL_NEWSTACK_BIN
    script_content = f"""from cryocat import tmana, cryomap, cryomotl

scores = cryomap.read(\"{scores_path}\")
angles = cryomap.read(\"{angles_path}\")

motl = tmana.scores_extract_particles(
    scores_map=scores,
    angles_map=angles,
    angles_list=\"{params['angle_file']}\",
    tomo_id={tomo_id},
    particle_diameter={params['particle_diameter']},
    object_id=None,
    scores_threshold={params['scores_threshold']},
    sigma_threshold={params['sigma_threshold']},
    cluster_size={params['cluster_size']},
    n_particles={params['n_particles']},
    output_path=None,
    output_type=\"emmotl\",
    angles_order=\"zxz\",
    symmetry=\"{params['symmetry']}\",
    angles_numbering=0,
)

rel_motl = cryomotl.RelionMotl(
    input_motl=motl.df,
    version=3.1, 
    pixel_size={pixel_size}, 
    binning=1, 
)

rel_motl.write_out(
    output_path=\"{output_star}\",
    use_original_entries=False,
    write_optics=False,
    pixel_size={pixel_size},
    binning=1,
)
"""
    
    cmd = f"python3 -c {shlex.quote(script_content)}"
    return cmd

def build_subtomo_extraction_command(params, jobs_per_gpu, gpu_devices):
    """Builds the command for the subtomogram extraction stage."""
    cmd = [
        "WarpTools", "ts_export_particles",
        "--settings", "warp_tiltseries.settings",
        "--input_directory", params["--input_directory"],
        "--input_pattern", params["--input_pattern"],
        "--input_processing", "warp_tiltseries",
        "--coords_angpix", str(params["--coords_angpix"]),
        "--output_star", params["--output_star"],
        "--output_angpix", str(params["--output_angpix"]),
        "--output_processing", params["--output_processing"],
        "--relative_output_paths",
        "--box", str(params["--box"]),
        "--diameter", str(params["--diameter"]),
        "--perdevice", str(jobs_per_gpu),
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    if params.get("3d", True):
        cmd.append("--3d")
        cmd.extend(["--max_missing_tilts", "999"])
    else:
        cmd.append("--2d")
    return cmd

def build_m_population_command(m_refine_params):
    """Builds the command for the m population, sources and species."""
    pop_cmd = [
        "MTools", "create_population",
         "--directory", m_refine_params["directory"],
         "--name", m_refine_params['population_name']
    ]

    source_cmds = []
    for source in m_refine_params["source_names"]:
        source_path = Path(cfg.base_dir) / source["dataset"] / "warp_tiltseries" / f"{source['name']}.source"
        if source_path.exists():
            source_cmds.append(["MTools", "add_source",
                "--population", f"{m_refine_params['directory']}/{m_refine_params['population_name']}.population",
                "--source", source_path
                ])
        else:
            source_cmds.append([
                "MTools", "create_source",
                "--name", source["name"],
                "--population", f"{m_refine_params['directory']}/{m_refine_params['population_name']}.population",
                "--processing_settings", Path(cfg.base_dir) / source["dataset"] / "warp_tiltseries.settings"
            ])
    
    species_cmds = []
    for species in m_refine_params["species"]:
        base_cmd = [
            "MTools", "create_species",
            "--population", f"{m_refine_params['directory']}/{m_refine_params['population_name']}.population",
            "--name", species["name"],
            "--diameter", "350",
            "--sym", "C1",
            "--temporal_samples", "1",
            "--half1", f"{m_refine_params['relion_folder']}/Refine3D/{species['job']}/run_half1_class001_unfil.mrc",
            "--half2", f"{m_refine_params['relion_folder']}/Refine3D/{species['job']}/run_half2_class001_unfil.mrc",
            "--mask", f"{m_refine_params['relion_folder']}/MaskCreate/{species['mask']}/mask.mrc",
            "--particles_relion", f"{m_refine_params['relion_folder']}/Refine3D/{species['job']}/run_data.star",
            "--angpix_coords", str(m_refine_params["input_angpix"]),
            "--angpix_resample", str(cfg.angpix),
            # "--angpix_resample", "2",
            "--lowpass", "15",
        ]
        if species["name"] in ["pf12", "pf13"]:
            base_cmd.extend([
                "--helical_units", "3",
                "--helical_twist", "29.88" if species["name"] == "pf12" else "27.69",
                "--helical_rise", "10.92" if species["name"] == "pf12" else "9.64",
                "--helical_height", "43",
            ])
        species_cmds.append(base_cmd)
    
    return [pop_cmd] + source_cmds + species_cmds

def build_m_refine_command(m_refine_params):
    """Builds the command for the m refine stage."""
    population = f"--population {m_refine_params['directory']}/{m_refine_params['population_name']}.population"
    device = "--perdevice_refine 1 --perdevice_preprocess 1 --perdevice_postprocess 1"
    cmds = [
        f"MCore {population} --iter 0 {device}",
        f"MCore {population} --iter 5 --refine_imagewarp 4x4 --refine_particles --ctf_defocus --ctf_defocusexhaustive {device}",
        f"MCore {population} --iter 5 --refine_imagewarp 4x4 --refine_particles --ctf_defocus {device}",
        f"MCore {population} --iter 5 --refine_imagewarp 4x4 --refine_particles --ctf_defocus {device}",
        f"EstimateWeights {population} --source m_full* --resolve_items",
        f"MCore {population} --iter 5 --refine_particles {device}",
        f"EstimateWeights {population} --source m_full* --resolve_frames",
        f"MCore {population} --iter 5 --refine_particles {device}",
        f"MCore {population} --iter 5 --refine_imagewarp 4x4 --refine_particles --ctf_defocus --refine_mag --ctf_cs --ctf_zernike3 --refine_stageangles {device}",
        # f"MTools resample_trajectories {population} --species {m_refine_params['directory']}/species/*/*.species --samples 3",
        # f"MCore {population} --iter 5 --refine_imagewarp 4x4 --refine_particles --ctf_defocus --refine_mag --ctf_cs --ctf_zernike3 --refine_stageangles {device}"
    ]
    return cmds
# --- Commands for run_pipeline.py ---

def build_frame_settings_command(frame_source_path, params):
    """Builds the command to create frame series settings."""
    cmd = [
        "WarpTools", "create_settings",
        "--folder_data", frame_source_path,
        "--folder_processing", "warp_frameseries",
        "--output", "warp_frameseries.settings",
        "--extension", params["extension"],
        "--angpix", str(cfg.angpix),
        "--exposure", str(cfg.dose),
    ]
    cmd.extend(params["extra_create_args"])
    return cmd

def build_tilt_settings_command(params):
    """Builds the command to create tilt series settings."""
    return [
        "WarpTools", "create_settings",
        "--output", "warp_tiltseries.settings",
        "--folder_processing", "warp_tiltseries",
        "--folder_data", "tomostar",
        "--extension", "*.tomostar",
        "--angpix", str(cfg.angpix),
        "--exposure", str(cfg.dose),
        "--tomo_dimensions", f"{params['original_x_y_size'][1]}x{params['original_x_y_size'][0]}x{cfg.thickness_pxl}"
    ]

def build_motion_ctf_command(params, jobs_per_gpu, gpu_devices):
    """Builds the command for motion correction and CTF estimation."""
    cmd = [
        "WarpTools", "fs_motion_and_ctf",
        "--settings", "warp_frameseries.settings",
        "--m_grid", f"1x1x{params['m_grid_frames']}",
        "--c_grid", "2x2x1",
        "--c_range_max", "7",
        "--c_defocus_max", "8",
        "--c_use_sum",
        "--out_averages",
        "--out_average_halves",
        "--perdevice", str(jobs_per_gpu)
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    return cmd

def build_histograms_command():
    """Builds the command to plot histograms."""
    return [
        "WarpTools", "filter_quality",
        "--settings", "warp_frameseries.settings",
        "--histograms"
    ]

def build_ts_import_command():
    """Builds the command to import tilt series metadata."""
    return [
        "WarpTools", "ts_import",
        "--mdocs", "mdocs",
        "--frameseries", "warp_frameseries",
        "--tilt_exposure", str(cfg.dose),
        "--dont_invert",
        "--override_axis", str(cfg.tilt_axis_angle),
        "--output", "tomostar"
    ]

def build_ts_etomo_patches_command(patch_size, jobs_per_gpu, gpu_devices):
    """Builds the command for eTomo patch-based alignment."""
    cmd = [
        "WarpTools", "ts_etomo_patches",
        "--settings", "warp_tiltseries.settings",
        "--angpix", str(cfg.angpix * cfg.FINAL_NEWSTACK_BIN),
        "--initial_axis", str(cfg.tilt_axis_angle),
        "--patch_size", str(patch_size),
        "--perdevice", str(jobs_per_gpu * 2)
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    return cmd

def build_ts_stack_command():
    """Builds the command to create tilt stacks."""
    return [
        "WarpTools", "ts_stack",
        "--settings", "warp_tiltseries.settings"
    ]

def build_import_align_command():
    """Builds the command to import improved alignments."""
    camera_map = {
        "Falcon4": cfg.angpix * cfg.FINAL_NEWSTACK_BIN,
        "K3": cfg.angpix
    }
    align_angpix = camera_map.get(cfg.camera_type)
    return [
        "WarpTools", "ts_import_alignments",
        "--settings", "warp_tiltseries.settings",
        "--alignments", "warp_tiltseries/tiltstack/",
        "--alignment_angpix", str(align_angpix),
    ]

def build_hand_check_command():
    """Builds the command to check defocus handedness."""
    return [
        "WarpTools", "ts_defocus_hand",
        "--settings", "warp_tiltseries.settings",
        "--check"
    ]

def build_hand_flip_command():
    """Builds the command to flip tomograms if handedness is wrong."""
    return [
        "WarpTools", "ts_defocus_hand",
        "--settings", "warp_tiltseries.settings",
        "--set_flip"
    ]

def build_ts_ctf_command(jobs_per_gpu, gpu_devices):
    """Builds the command to estimate tilt series CTF."""
    cmd = [
        "WarpTools", "ts_ctf",
        "--settings", "warp_tiltseries.settings",
        "--range_high", "7",
        "--defocus_max", "8",
        "--perdevice", str(jobs_per_gpu)
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    return cmd
