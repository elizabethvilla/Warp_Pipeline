#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, os.getcwd())
import etomo_align
import config as cfg
from pipeline_utils import (
    run_command, 
    update_xml_files_from_com, 
    reorganize_falcon4_data, 
    calculate_patch_size, 
    prepare_gain_reference
)
from command_builder import (
    build_frame_settings_command,
    build_tilt_settings_command,
    build_motion_ctf_command,
    build_histograms_command,
    build_ts_import_command,
    build_ts_etomo_patches_command,
    build_ts_stack_command,
    build_import_align_command,
    build_hand_check_command,
    build_hand_flip_command,
    build_ts_ctf_command,
)

def run_preprocess(dataset_dir: Path, logs_dir: Path, params: dict):
    """Runs the preprocessing stage."""
    logging.info("Starting preprocessing stage...")
    
    logging.info("Linking mdocs files...")
    (dataset_dir / "mdocs").mkdir(exist_ok=True)
    
    mdoc_source_path = Path(cfg.raw_directory) / cfg.dataset_name / cfg.mdoc_folder
    # for mdoc_file in mdoc_source_path.glob(f"{cfg.tomo_match_string}*ts*.mrc.mdoc"):
    for mdoc_file in mdoc_source_path.glob(f"{cfg.tomo_match_string}*.mdoc"):
        # dest_file = dataset_dir / "mdocs" / mdoc_file.name.replace(".mrc.", ".")
        dest_file = dataset_dir / "mdocs" / mdoc_file.name
        if not dest_file.exists():
            dest_file.symlink_to(mdoc_file)

    frame_source_path = Path(cfg.raw_directory) / cfg.dataset_name / cfg.frame_folder

    if cfg.camera_type == "K3":
        logging.info("Preparing gain reference for K3 camera...")
        gain_ref_link = prepare_gain_reference(cfg, frame_source_path)
        if not gain_ref_link:
            logging.critical("Gain reference preparation failed. Please check logs. Aborting preprocessing.")
            return
        params["extra_create_args"].extend(["--gain_path", str(gain_ref_link.resolve())])

    logging.info("Creating frame series settings...")
    cmd_frame_settings = build_frame_settings_command(frame_source_path, params)
    run_command(cmd_frame_settings, logs_dir / "frame_settings.log", cwd=dataset_dir)

    logging.info("Creating tilt series settings...")
    (dataset_dir / "tomostar").mkdir(exist_ok=True)
    cmd_tilt_settings = build_tilt_settings_command(params)
    run_command(cmd_tilt_settings, logs_dir / "tilt_settings.log", cwd=dataset_dir)

    logging.info("Running frame series motion and CTF estimation...")
    cmd_motion_ctf = build_motion_ctf_command(params, cfg.jobs_per_gpu, cfg.gpu_devices)
    run_command(cmd_motion_ctf, logs_dir / "motion_ctf.log", cwd=dataset_dir)

    logging.info("Plotting histograms of 2D processing metrics...")
    cmd_histograms = build_histograms_command()
    run_command(cmd_histograms, logs_dir / "histograms.log", cwd=dataset_dir)

    logging.info("Preprocessing stage completed.")

def run_ts_import(dataset_dir: Path, logs_dir: Path):
    """Runs the tilt series import stage."""
    logging.info("Importing tilt series metadata...")
    
    cmd_ts_import = build_ts_import_command()
    run_command(cmd_ts_import, logs_dir / "tomostar.log", cwd=dataset_dir)
    
    logging.info("Tilt series import stage completed.")

def run_builtin_etomo(dataset_dir: Path, logs_dir: Path):
    """Runs the eTomo alignment stage with a patched environment."""
    logging.info("Starting Patched eTomo alignment stage...")

    run_ts_import(dataset_dir, logs_dir)

    pipeline_dir = Path(__file__).parent.resolve()
    wrapper_dir = pipeline_dir / 'imod_wrappers'
    if not wrapper_dir.is_dir():
        logging.error(f"IMOD wrapper directory not found at: {wrapper_dir}")
        return

    env = os.environ.copy()
    env['PATH'] = f"{wrapper_dir}{os.pathsep}{env.get('PATH', '')}"   

    patch_size = calculate_patch_size(cfg)

    cmd_ts_etomo = build_ts_etomo_patches_command(patch_size, cfg.jobs_per_gpu, cfg.gpu_devices)
    run_command(cmd_ts_etomo, logs_dir / "etomo_patches.log", cwd=dataset_dir, env=env)

    logging.info("Patched eTomo test stage completed.")

def run_custom_etomo(dataset_dir: Path, logs_dir: Path):
    """Runs the eTomo alignment stage with proper directory management."""
    logging.info("Starting eTomo alignment stage...")
    
    run_ts_import(dataset_dir, logs_dir)

    logging.info("Making tilt stacks...")
    cmd_ts_stack = build_ts_stack_command()
    run_command(cmd_ts_stack, logs_dir / "tiltstack.log", cwd=dataset_dir)

    original_dir = dataset_dir.resolve()
    etomo_dir = original_dir / "warp_tiltseries" / "tiltstack"

    if not etomo_dir.is_dir():
        logging.error(f"eTomo directory not found at: {etomo_dir}")
        logging.error("Please ensure the preprocessing stage was run successfully.")
        return

    try:
        logging.info(f"Changing directory to {etomo_dir}")
        os.chdir(etomo_dir)
        etomo_align.run_alignment()
    finally:
        logging.info(f"Returning to directory: {original_dir}")
        os.chdir(original_dir)
        
    logging.info("eTomo alignment stage completed.")

def optimize_etomo(dataset_dir: Path, logs_dir: Path):
    """Runs the customized eTomo optimization stage."""
    logging.info("Starting eTomo optimization stage...")
    
    etomo_dir = dataset_dir / "warp_tiltseries" / "tiltstack"
    if not etomo_dir.is_dir():
        logging.error(f"eTomo directory not found at: {etomo_dir}")
        logging.error("Please ensure the preprocessing stage was run successfully.")
        return

    script_path = Path(__file__).parent.resolve() / 'etomo_optimize.py'
    log_path = logs_dir / 'etomo_optimization.log'

    command = [
        sys.executable, str(script_path), 
        '--tiltstack_dir', str(etomo_dir),
        '--main_logs_dir', str(logs_dir)
    ]

    try:
        run_command(command, log_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"eTomo optimization script failed: {e}")
        logging.error(f"Check the log for details: {log_path}")
        
    logging.info("eTomo optimization stage completed.")

def run_postprocess(dataset_dir: Path, logs_dir: Path):
    """Runs the post-processing stage."""
    logging.info("Starting post-processing stage...")
    
    logging.info("Extracting RMD and saving table...")
    rmd_error_log = logs_dir / "eTomo_RMD_error.txt"
    if rmd_error_log.exists():
        rmd_error_log.rename(str(rmd_error_log) + "~")
    
    with rmd_error_log.open("w") as f:
        for d in (dataset_dir / "warp_tiltseries/tiltstack").glob(f"{cfg.tomo_match_string}*"):
            align_log = d / "align_clean.log"
            if align_log.exists():
                with align_log.open("r") as log_file:
                    for line in log_file:
                        if "Residual error weighted mean" in line:
                            new_rm = line.split()[-2]
                            f.write(f"{d.name}\t{new_rm}\n")
                            break

    logging.info("Importing improved alignments...")
    cmd_import_align = build_import_align_command()
    run_command(cmd_import_align, logs_dir / "import_align.log", cwd=dataset_dir)

    logging.info("Checking defocus handedness...")
    cmd_hand_check = build_hand_check_command()
    result = subprocess.run(cmd_hand_check, check=True, capture_output=True, text=True, cwd=dataset_dir)
    (logs_dir / "handness.log").write_text(result.stdout)
    
    if "no flip" not in result.stdout:
        logging.info("Flipping tomograms...")
        cmd_hand_flip = build_hand_flip_command()
        run_command(cmd_hand_flip, logs_dir / "flipped.log", cwd=dataset_dir)
    else:
        logging.info("No need to flip tomograms.")

    logging.info("Estimating tilt series CTF...")
    cmd_ts_ctf = build_ts_ctf_command(cfg.jobs_per_gpu, cfg.gpu_devices)
    run_command(cmd_ts_ctf, logs_dir / "tomo_ctf.log", cwd=dataset_dir)

    # --- XML Parsing with isolated logging ---
    logging.info("Parsing XML files to remove bad tilts...")
    xml_log_path = logs_dir / 'xml_parsing.log'
    xml_handler = logging.FileHandler(xml_log_path, mode='w')
    xml_handler.setFormatter(logging.getLogger().handlers[0].formatter)

    xml_logger = logging.getLogger("xml_updater")
    xml_logger.addHandler(xml_handler)
    xml_logger.propagate = False

    warp_tiltseries_dir = dataset_dir / "warp_tiltseries"
    xml_backup_dir = warp_tiltseries_dir / "xml_backup"
    xml_backup_dir.mkdir(exist_ok=True)
    
    for xml_file in warp_tiltseries_dir.glob("*.xml"):
        if not (xml_backup_dir / xml_file.name).exists():
            shutil.copy(xml_file, xml_backup_dir / xml_file.name)
    
    for xml_file in xml_backup_dir.glob("*.xml"):
        shutil.copy(xml_file, warp_tiltseries_dir / xml_file.name)
    
    update_xml_files_from_com(warp_tiltseries_dir)
    
    # Clean up the handler and restore propagation
    xml_logger.removeHandler(xml_handler)
    xml_logger.propagate = True
    xml_handler.close()
    logging.info(f"XML parsing logs saved to {xml_log_path}")

    # --- Deconvolution with non-verbose logging ---
    xml_files_to_process = list(warp_tiltseries_dir.glob("*.xml"))
    logging.info(f"Applying deconvolution for {len(xml_files_to_process)} tomograms...")
    tomogram_logs_dir = logs_dir / 'tomograms'

    for xml_file in xml_files_to_process:
        tomo_name = xml_file.stem
        defocus = ""
        with xml_file.open("r") as f:
            for line in f:
                if 'Defocus" Val' in line:
                    defocus = line.split('"')[3]
                    break
        
        if not defocus:
            logging.warning(f"Defocus not found for {tomo_name}, skipping deconvolution.")
            continue
        
        tomo_dir = warp_tiltseries_dir / "tiltstack" / tomo_name
        rec_file = f"{tomo_name}_rot_flipz.mrc"
        mrc_file = f"{tomo_name}_rot_flipz_dev.mrc"

        if not (tomo_dir / rec_file).exists():
            logging.warning(f"Input file not found, skipping deconvolution for {tomo_name}: {tomo_dir/rec_file}")
            continue
        
        command_string = (
            f"module unload imod; "
            f"module load imod/5.0.1-beta; "
            f"reducefiltvol -i {rec_file} -o {mrc_file} -dec 0.5 -def {defocus}"
        )
        
        deconv_log_dir = tomogram_logs_dir / tomo_name
        deconv_log_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [command_string],
            deconv_log_dir / "deconvolution.log",
            cwd=tomo_dir,
            shell=True,
            verbose=False
        )

        for f in tomo_dir.glob("*~"):
            f.unlink()

    logging.info("Post-processing stage completed.")

def main():
    """Main function to drive the pipeline."""
    parser = argparse.ArgumentParser(description="A flexible pipeline for cryo-ET data processing.")
    parser.add_argument(
        '--stage',
        type=str,
        choices=['preprocess', 'etomo', 'optimize', 'postprocess', 'all'],
        default='all',
        help="Which stage of the pipeline to run."
    )
    args = parser.parse_args()

    dataset_dir = Path.cwd()
    if dataset_dir.name != cfg.dataset_name:
        logging.warning(f"⚠️  Current directory name '{dataset_dir.name}' matches config '{cfg.dataset_name}'?")

    logs_dir = dataset_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file_path = logs_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Processing will run in: {dataset_dir}")
    logging.info(f"Main log file for this run is: {log_file_path.resolve()}")
    logging.info(f"Loaded configuration from: {cfg.__file__}")

    if args.stage in ['all', 'preprocess'] and cfg.camera_type == "Falcon4":
        logging.info("Falcon4 camera type detected. Checking if data reorganization is needed...")
        try:
            reorganize_falcon4_data(cfg, logs_dir)
            logging.info("Data reorganization completed successfully.")
        except Exception as e:
            logging.critical(f"Data reorganization failed: {e}", exc_info=True)
            logging.critical("Cannot proceed with the pipeline. Please check the configuration and source directory.")
            sys.exit(1)
            
    if args.stage in ['all', 'preprocess']:
        run_preprocess(dataset_dir, logs_dir, cfg.pipeline_params)
    if args.stage in ['all', 'etomo']:
        run_builtin_etomo(dataset_dir, logs_dir) if cfg.camera_type == "Falcon4" else run_custom_etomo(dataset_dir, logs_dir)
    if args.stage in ['all', 'optimize']:
        optimize_etomo(dataset_dir, logs_dir)
    if args.stage in ['all', 'postprocess']:
        run_postprocess(dataset_dir, logs_dir)

    logging.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()
