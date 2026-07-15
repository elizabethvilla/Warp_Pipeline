#!/usr/bin/env python3

import argparse
import logging
import subprocess
import sys
import os
import re
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.getcwd())
import config as cfg
from pipeline_utils import run_command, filter_star_file, run_parallel_tasks, load_tomo_list, detect_gpu_arch
from command_builder import (
    build_m_refine_command,
    build_m_population_command,
    build_reconstruction_command,
    build_isonet_commands,
    build_isonet2_commands,
    build_cryolo_commands,
    build_template_match_command,
    build_subtomo_extraction_command,
    build_gapstop_wedge_command,
    build_gapstop_result_command
)

try:
    from star_handler.modules.processors.relion2cbox import Relion2CboxProcessor
except ImportError:
    logging.warning("Could not import star_handler. Automatic Relion2Cbox processing will be disabled.")
    Relion2CboxProcessor = None

def reconstruction(log_file_path: Path):
    """Runs the final reconstruction and packaging stage for Windows compatibility."""
    logging.info("Starting final reconstruction and packaging stage...")

    win_dir = Path("forWindows_frames")
    win_dir.mkdir(exist_ok=True)
    logging.info(f"{win_dir.resolve()} is ready for packaging.")

    logging.info("Running WarpTools ts_reconstruct...")
    env = os.environ.copy()
    env['WARP_FORCE_MRC_FLOAT32'] = '1'
    cmd_reconstruct = build_reconstruction_command(cfg.jobs_per_gpu, cfg.gpu_devices)
    run_command(cmd_reconstruct, log_file_path, env=env)

    logging.info(f"Linking result files into {win_dir}...")
    
    warp_frameseries_dir = Path("warp_frameseries")
    warp_tiltseries_dir = Path("warp_tiltseries")
    tomostar_dir = Path("tomostar")

    link_pairs = {
        "average": warp_frameseries_dir / "average",
        "reconstruction": warp_tiltseries_dir / "reconstruction",
    }
    for link_name, target_path in link_pairs.items():
        dest_link = win_dir / link_name
        if not dest_link.exists() and target_path.exists():
            relative_target = os.path.relpath(target_path.resolve(), win_dir.resolve())
            dest_link.symlink_to(relative_target)

    for xml_file in warp_tiltseries_dir.glob("*.xml"):
        dest_link = win_dir / xml_file.name
        if not dest_link.exists():
            relative_target = os.path.relpath(xml_file.resolve(), win_dir.resolve())
            dest_link.symlink_to(relative_target)
            
    for star_file in tomostar_dir.glob("*.tomostar"):
        dest_link = win_dir / star_file.name
        if not dest_link.exists():
            relative_target = os.path.relpath(star_file.resolve(), win_dir.resolve())
            dest_link.symlink_to(relative_target)

    logging.info("Reconstruction and packaging stage completed.")

def isonet(log_file_path: Path):
    """Run the ISONet stage of the pipeline."""
    logging.info("Starting ISONet stage...")
    tomo_list = load_tomo_list()

    tomo_folder = "tomoset"
    isonet_dir = Path("isonet")
    cmd_log_dir = isonet_dir / "logs"
    Path(cmd_log_dir).mkdir(parents=True, exist_ok=True)
    Path(isonet_dir / tomo_folder).mkdir(parents=True, exist_ok=True)

    for tomo in tomo_list:
        source = Path("warp_tiltseries/reconstruction/deconv_flip/").resolve()
        target = isonet_dir / tomo_folder / f"{tomo}.mrc"
        source_files = list(source.glob(f"{tomo}*_10.00Apx.mrc"))
        if not source_files:
            logging.warning(f"No _dev.mrc file found for tomogram {tomo}")
            continue
        if not target.exists():
            target.symlink_to(source_files[0])

    commands = build_isonet_commands(cfg.isonet_params, cfg.gpu_devices, tomo_folder)
    
    total_steps = len(commands)
    for i, cmd in enumerate(commands, 1):
        logging.info(f"--- Starting ISONet step [{i}/{total_steps}]: {cmd.split()[1]} ---")
        run_command(cmd, cmd_log_dir / f"step_{i}.log", cwd=isonet_dir, module_load='isonet')
    
    logging.info("--- All ISONet steps completed successfully. ---")

def isonet2(log_file_path: Path):
    """Run the ISONet2 stage of the pipeline."""
    logging.info("Starting ISONet2 stage...")
    tomo_list = load_tomo_list()

    isonet_dir = Path("isonet2")
    cmd_log_dir = isonet_dir / "logs"
    for sub in ["even", "odd", "logs"]:
        (isonet_dir / sub).mkdir(parents=True, exist_ok=True)

    for tomo in tomo_list:
        for half in ["even", "odd"]:
            src_dir = Path(f"warp_tiltseries/reconstruction/{half}/")
            src_files = list(src_dir.glob(f"{tomo}*.mrc"))
            if not src_files:
                logging.warning(f"No {half} half-map found for {tomo}")
                continue
            dst = isonet_dir / half / src_files[0].name
            if not dst.exists():
                dst.symlink_to(src_files[0].resolve())

    defocus_list = []
    for tomo in sorted(tomo_list):
        xml_path = Path(f"warp_tiltseries/{tomo}.xml")
        match = re.search(r'<Param Name="Defocus" Value="([0-9.]+)"',
                          xml_path.read_text()) if xml_path.exists() else None
        if match:
            defocus_list.append(round(float(match.group(1)) * 10000))
        else:
            logging.warning(f"Defocus not found for {tomo}, using placeholder 10000")
            defocus_list.append(10000)

    logging.info(f"Extracted defocus values: {defocus_list}")

    commands = build_isonet2_commands(cfg.isonet2_params, cfg.gpu_devices, defocus_list)

    gpu_arch = detect_gpu_arch(cfg.BLACKWELL_HOSTNAMES)
    logging.info(f"ISONet2 GPU arch: {gpu_arch} (host={socket.gethostname()})")

    total_steps = len(commands)
    for i, cmd in enumerate(commands, 1):
        logging.info(f"--- Starting ISONet2 step [{i}/{total_steps}]: {cmd.split()[1]} ---")
        if gpu_arch == 'blackwell':
            run_command(cmd, cmd_log_dir / f"step_{i}.log",
                        cwd=isonet_dir, conda_env='isonet2_blackwell_test')
        else:
            run_command(cmd, cmd_log_dir / f"step_{i}.log",
                        cwd=isonet_dir, module_load='isonet2/2.0.1b-dev')

    logging.info("--- All ISONet2 steps completed successfully. ---")

def cryolo(log_file_path: Path):
    """Run the Cryolo stage of the pipeline."""
    cryolo_dir = Path("cryolo")    
    if not cryolo_dir.exists():
        if cfg.cryolo_params['prep']["enable"] and Relion2CboxProcessor:
            logging.info("Cryolo directory not found. Auto-running Relion2CboxProcessor...")
            try:
                processor = Relion2CboxProcessor(
                    star_file=cfg.cryolo_params['prep']['star_file'],
                    bin_factor=cfg.cryolo_params['prep']['bin_factor']
                )
                processor.process()
                logging.info("Relion2CboxProcessor completed successfully.")
            except Exception as e:
                logging.error(f"Auto-prep failed: {e}")
                sys.exit(1)
        else:
            reason = "Auto-prep is disabled" if not cfg.cryolo_params['prep']["enable"] else "star_handler not imported"
            logging.error(f"Cryolo directory missing and cannot auto-fix ({reason}).")
            logging.error("Please run 'star-handler process-relion2cryolo' manually or check your config.")
            sys.exit(1)

    list_file = Path('ribo_list_final.txt')
    if not list_file.exists():
        logging.error(f"tomogram list file does not exist in the current directory: {list_file.resolve()}")
        sys.exit(1)

    commands, output_dir = build_cryolo_commands(cfg.cryolo_params, cfg.gpu_devices)
    
    cmd_log_dir = cryolo_dir / "logs"
    Path(cmd_log_dir).mkdir(parents=True, exist_ok=True)
    total_steps = len(commands)
    for i, cmd in enumerate(commands, 1):
        step_name = cmd.split()[4] 
        logging.info(f"--- Starting CryoLo step [{i}/{total_steps}]: {step_name} ---")
        run_command(cmd, cmd_log_dir / f"step_{i}.log", cwd=cryolo_dir, module_load="cryolo")
    
    with open(list_file, 'r') as f:
        to_star_log_dir = cmd_log_dir / "to_star"
        Path(to_star_log_dir).mkdir(parents=True, exist_ok=True)

        for line in f.readlines():
            tomo, start_str, end_str, _ = line.strip().split()
            logging.info(f"Processing tomogram: {tomo}")

            coords_file = f"COORDS/{tomo}.coords"
            raw_star_dir = cryolo_dir / output_dir / "STAR" / tomo
            raw_star_dir.mkdir(parents=True, exist_ok=True)
            raw_star_file = raw_star_dir / "particles_warp.star"

            cmd_coords = [
                "cryolo_boxmanager_tools.py", "coords2star",
                "-i", str(cryolo_dir / output_dir / coords_file),
                "-o", str(raw_star_dir),
                # "--apix", str(cfg.angpix * cfg.FINAL_NEWSTACK_BIN)
                "--apix", "10"
            ]
            run_command(cmd_coords, to_star_log_dir / f"{tomo}.log", module_load="cryolo")

            filtered_star_file = cryolo_dir / output_dir / "STAR" / f"{tomo}.star"
            logging.info(f"Filtering {raw_star_file} to {filtered_star_file} with range {start_str}-{end_str}")
            
            try:
                z_range = (float(start_str), float(end_str))
                filter_star_file(raw_star_file, filtered_star_file, z_range)
            except ValueError:
                logging.error(f"Invalid range for tomogram {tomo}: {start_str}-{end_str}")

    logging.info("--- Handing over to subtomo_extraction ---")
    
    subtomo_params = {
        "--input_directory": str(cryolo_dir / output_dir / "STAR"),
        "--input_pattern": "*.star",
        # "--coords_angpix": str(cfg.angpix * cfg.FINAL_NEWSTACK_BIN),
        "--coords_angpix": "10",
        "--output_star": f"relion32_cryolo_expand/cryolo_{output_dir}.star",
        "--output_angpix": str(cfg.angpix * cfg.FINAL_NEWSTACK_BIN / 2),
        "--output_processing": "relion32_cryolo_expand/",
        "--box": "72",
        "--diameter": "350",
        "3d": True
    }

    original_params = cfg.subtomo_params
    cfg.subtomo_params = subtomo_params 
    subtomo_extraction(log_file_path)
    cfg.subtomo_params = original_params
    logging.info("--- Subtomo extraction after CryoLo completed. ---")

def _findsection_worker(tomo_dir: Path) -> Tuple[str, Optional[float], Optional[float]]:
    """Run findsection on one tomogram directory and return (name, z_start, z_end)."""
    tomo_name = tomo_dir.name
    z_files = sorted(tomo_dir.glob("*z.mrc"))
    if not z_files:
        return tomo_name, None, None
    try:
        cmd = f"module load imod; findsection -scales 4 -size 32,32,1 {z_files[0]}"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=300
        )
        for line in result.stdout.splitlines():
            if "Abso" in line:
                parts = line.split()
                return tomo_name, float(parts[8]), float(parts[9])
        return tomo_name, None, None
    except Exception:
        return tomo_name, None, None


def _run_findsection(log_dir: Path) -> None:
    """Run findsection in parallel across all tomograms, write ribo_list.txt and ribo_list_failed.txt."""
    tiltstack_dir = Path("warp_tiltseries/tiltstack")
    if not tiltstack_dir.exists():
        logging.error(f"tiltstack directory not found: {tiltstack_dir.resolve()}")
        return

    tomo_dirs = sorted([d for d in tiltstack_dir.iterdir() if d.is_dir()])
    if not tomo_dirs:
        logging.warning("No tomogram directories found in tiltstack.")
        return

    logging.info(f"--- Running findsection on {len(tomo_dirs)} tomograms ---")
    results = run_parallel_tasks(
        _findsection_worker,
        tomo_dirs,
        logger=logging.getLogger(__name__),
    )

    success, failed = [], []
    for tomo_name, z_start, z_end in results:
        if z_start is not None:
            success.append((tomo_name, z_start, z_end))
        else:
            failed.append(tomo_name)

    with open("ribo_list.txt", "w") as f:
        for tomo_name, z_start, z_end in success:
            f.write(f"{tomo_name}\t{z_start}\t{z_end}\t0\n")

    with open("ribo_list_failed.txt", "w") as f:
        for tomo_name in failed:
            f.write(f"{tomo_name}\n")

    logging.info(
        f"findsection done: {len(success)} succeeded, {len(failed)} failed. "
        f"Results: ribo_list.txt | Failed: ribo_list_failed.txt"
    )
    logging.info(
        ">>> 3DTM is now running. Use this time to review ribo_list.txt, "
        "add missing entries from ribo_list_failed.txt, and save as ribo_list_final.txt. <<<"
    )


def _run_starhandler_scale(log_dir: Path) -> None:
    """Run star-handler process-3DTM2relion (scale step, no ribo_list_final.txt present)."""
    matching_dir = Path("warp_tiltseries/matching")
    if not matching_dir.exists():
        logging.warning(f"matching directory not found: {matching_dir.resolve()}. Skipping star-handler scale.")
        return

    # star-handler goes to filter mode if ribo_list_final.txt exists — hide it temporarily
    list_file = Path("ribo_list_final.txt")
    list_file_tmp = Path("ribo_list_final.txt.bak")
    hidden = list_file.exists()
    if hidden:
        list_file.rename(list_file_tmp)
        logging.info("Temporarily hiding ribo_list_final.txt to force scale mode.")

    log_path = log_dir / "3DTM_starhandler_scale.log"
    try:
        run_command(
            ["star-handler", "process-3DTM2relion", "-d", str(matching_dir)],
            log_path,
        )
        logging.info("--- star-handler scale completed → warp_tiltseries/matching/scaled/ ---")
        logging.info(
            ">>> Complete ribo_list_final.txt (Z bounds + CC threshold), "
            "then run: python run_appendix.py --stage filter_3dtm <<<"
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        logging.warning(
            "star-handler not available or failed. Run manually: "
            f"star-handler process-3DTM2relion -d {matching_dir.resolve()}"
        )
    finally:
        if hidden:
            list_file_tmp.rename(list_file)


def template_match_3D(log_file_path: Path):
    """Run the 3D template matching stage of the pipeline."""
    # Step 1: parallel findsection → ribo_list.txt (runs before the slow WarpTools step)
    _run_findsection(log_file_path.parent)

    # Step 2: Warp template matching
    template_path = Path(cfg.template_matching_params['template_path'])
    if not template_path.exists():
        logging.error(f"Template file does not exist: {template_path.resolve()}")
        sys.exit(1)

    list_file = Path(cfg.template_matching_params['input_data'])
    if not list_file.exists():
        logging.warning(f"no list file available: {list_file.resolve()}. Running with full tomoset.")

    env = os.environ.copy()
    cmd_template_match = build_template_match_command(
        cfg.template_matching_params, cfg.jobs_per_gpu, cfg.gpu_devices
    )

    logging.info(f"--- Starting Warp 3D template matching ---")
    run_command(cmd_template_match, log_file_path, env=env, module_load="warp/2.0.0dev39")
    logging.info("--- WarpTools ts_template_match completed. ---")

    # Step 3: star-handler scale → matching/scaled/ (requires 3DTM star files)
    _run_starhandler_scale(log_file_path.parent)


def filter_3dtm(log_file_path: Path):
    """Filter 3DTM picks using ribo_list_final.txt (star-handler step 2)."""
    list_file = Path("ribo_list_final.txt")
    if not list_file.exists():
        logging.error(
            f"ribo_list_final.txt not found at {list_file.resolve()}. "
            "Please annotate ribo_list.txt and save it as ribo_list_final.txt first."
        )
        sys.exit(1)

    matching_dir = Path("warp_tiltseries/matching")
    if not matching_dir.exists():
        logging.error(f"matching directory not found: {matching_dir.resolve()}")
        sys.exit(1)

    logging.info("--- Running star-handler filter (process-3DTM2relion with ribo_list_final.txt) ---")
    try:
        run_command(
            ["star-handler", "process-3DTM2relion", "-d", str(matching_dir)],
            log_file_path,
        )
        logging.info("--- filter_3dtm completed → warp_tiltseries/matching/filtered/ ---")
    except (FileNotFoundError, subprocess.CalledProcessError):
        logging.error(
            "star-handler not available or failed. Run manually: "
            f"star-handler process-3DTM2relion -d {matching_dir.resolve()}"
        )
        sys.exit(1)


def _gapstop_wedge_worker(tomo, tomo_id, wedge_dir, env, wedge_log_dir):
    wedge_star_path = wedge_dir / f"wedge_{tomo}.star"
    wedge_log_path = wedge_log_dir / f"{tomo}.log"
    if wedge_star_path.exists():
        return f"Skipped {tomo}: wedge file exists"

    cmd_wedge = build_gapstop_wedge_command(cfg.gapstop_params, tomo, tomo_id)
    run_command(cmd_wedge, wedge_log_path, cwd=wedge_dir, env=env, module_load="gapstop/0.3")
    return f"Completed {tomo}"


def _gapstop_result_worker(tomo, tomo_id, result_log_dir):
    """Extract particles from GapStop template matching results."""
    output_star_path = Path("results") / f"{tomo}.star"
    result_log_path = result_log_dir / f"{tomo}.log"
    gapstop_path = result_log_dir.parent.parent
    if output_star_path.exists():
        return f"Skipped {tomo}: result star exists"
        
    cmd_result = build_gapstop_result_command(
        cfg.gapstop_params,
        tomo_id,
        str(output_star_path)
    )
    run_command(cmd_result, result_log_path, cwd=gapstop_path, module_load="gapstop/0.3")

    target_file = gapstop_path / output_star_path
    if target_file.exists():
        lines = target_file.read_text().splitlines()
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) > 5:
                parts[0] = f"{tomo}.tomostar"
                lines[i] = '\t'.join(parts)
        target_file.write_text('\n'.join(lines) + '\n')

    return f"Completed {tomo}"


def template_match_gapstop(log_file_path: Path):
    """Run the 3D template matching using gapstop.
    0. Make sure eTomo and Warp reconstruction shares the same coordinate
    1. create models and maps -- one for all tomograms
    2. create _anglist_name file -- also one for all tomograms
    0-2 are all done outof the box, and required files will be checked before running. Throw error if missing.
    3. create wedge.star -- one file for each tomogram. _tomo_num matches, _defocus is unique. Use _ = wedgeutils.create_wedge_list_sg() to create each one
    4. create tm_param.star -- one row for each tomogram. _tomo_name _tomo_num _wedgelist_name _smap_name _omap_name are unique
    5. test run
    6. use tmana.scores_extract_particles() to check on the result
    """
    gapstop_dir = Path("gapstop")
    wedge_dir = gapstop_dir / "wedge"
    wedge_dir.mkdir(parents=True, exist_ok=True)
    output_dir = gapstop_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    required_files = [
        gapstop_dir / cfg.gapstop_params['template'],
        gapstop_dir / cfg.gapstop_params['mask'],
        gapstop_dir / cfg.gapstop_params['angle_file'],
    ]
    for f in required_files:
        if not f.exists():
            logging.error(f"Required file not found: {f.resolve()}")
            sys.exit(1)

    tomo_list = load_tomo_list()

    env = os.environ.copy()
    logging.info(f"--- Starting gapstop 3D template matching ---")
    wedge_log_dir = gapstop_dir / "logs" / "wedge"
    wedge_log_dir.mkdir(parents=True, exist_ok=True)

    task_args = []
    for i, tomo in enumerate(tomo_list, 1):
        wedge_star_path = wedge_dir / f"wedge_{tomo}.star"
        if not wedge_star_path.exists():
            task_args.append((tomo, i, wedge_dir, env, wedge_log_dir))
        else:
            logging.info(f"Wedge star already exists for {tomo}, skipping: {wedge_star_path.resolve()}")

    if task_args:
        results = run_parallel_tasks(
            _gapstop_wedge_worker,
            task_args,
            num_workers=cfg.gapstop_workers,
            logger=logging.getLogger(__name__),
            use_starmap=True,
        )
        for res in results:
            logging.info(res)
    else:
        logging.info("All wedge files already exist, skipping parallel wedge generation.")

    tm_param_path = gapstop_dir / "tm_param.star"
    
    template_path = Path(__file__).parent / "tm_param_template.star"
    with open(template_path, 'r') as f:
        template = f.read()
    
    rootdir = gapstop_dir.resolve()
    outputdir = "tm_outputs/"
    vol_ext = ".mrc"
    tmpl_name = Path(cfg.gapstop_params['template']).name
    mask_name = Path(cfg.gapstop_params['mask']).name
    anglist_name = Path(cfg.gapstop_params['angle_file']).name
    symmetry = "C1"
    anglist_order = "zxz"
    lp_rad = 16
    hp_rad = 1
    binning = 8
    tiling = "new"
    
    tomogram_rows = []
    for i, tomo in enumerate(tomo_list, 1):
        tomo_name = f"../warp_tiltseries/reconstruction/{tomo}_{cfg.angpix * cfg.FINAL_NEWSTACK_BIN}Apx.mrc"
        wedgelist_name = f"wedge/wedge_{tomo}.star"
        smap_name = f"scores"
        omap_name = f"angles"
        row = f"{rootdir} {outputdir} {vol_ext} {tomo_name} {i} {wedgelist_name} {tmpl_name} {mask_name} {symmetry} {anglist_order} {anglist_name} {smap_name} {omap_name} {lp_rad} {hp_rad} {binning} {tiling}"
        tomogram_rows.append(row)
    
    tm_param_content = template.replace("{TOMOGRAM_ROWS}", "\n".join(tomogram_rows))
    with open(tm_param_path, 'w') as f:
        f.write(tm_param_content)
    
    logging.info(f"Generated tm_param.star with {len(tomo_list)} tomograms: {tm_param_path.resolve()}")

    gapstop_cmd = ["gapstop", "run_tm", "-n", "8", "tm_param.star"]
    run_command(gapstop_cmd, log_file_path, cwd=gapstop_dir, env=env, module_load="gapstop/0.3")

    logging.info("--- Starting parallel particle extraction from GapStop results ---")
    result_log_dir = gapstop_dir / "logs" / "result"
    result_log_dir.mkdir(parents=True, exist_ok=True)
    
    result_task_args = []
    for i, tomo in enumerate(tomo_list, 1):
        output_star = gapstop_dir / output_dir / f"{tomo}.star"
        if not output_star.exists():
            result_task_args.append((tomo, i, result_log_dir))
        else:
            logging.info(f"Result star already exists for {tomo}, skipping: {output_star.resolve()}")
    
    if result_task_args:
        result_results = run_parallel_tasks(
            _gapstop_result_worker,
            result_task_args,
            num_workers=cfg.gapstop_workers//12,
            logger=logging.getLogger(__name__),
            use_starmap=True,
        )
        for res in result_results:
            logging.info(res)
    else:
        logging.info("All result star files already exist, skipping particle extraction.")
    
    logging.info("--- gapstop 3D template matching completed. ---")  



def subtomo_extraction(log_file_path: Path):
    """Run the particle extraction stage of the pipeline."""
    env = os.environ.copy()
    if cfg.subtomo_params.get("3d", True):
        env['WARP_FORCE_MRC_FLOAT32'] = '1'
    cmd_export = build_subtomo_extraction_command(
        cfg.subtomo_params, cfg.jobs_per_gpu, cfg.gpu_devices
    )

    logging.info("--- Starting WarpTools ts_export_particles ---")
    run_command(cmd_export, log_file_path, env=env, module_load="warp/2.0.0dev39")
    logging.info("--- WarpTools ts_export_particles completed. ---")


def m_refinement(log_file_path: Path):
    """Run the M refinement stage of the pipeline."""
    m_dir = Path("ribo_m")
    cmd_log_dir = m_dir / cfg.m_refine_params['directory'] / "logs"
    env = os.environ.copy()

    logging.info("--- Preparing population ---")
    prep_cmds = build_m_population_command(cfg.m_refine_params)
    total_steps = len(prep_cmds)
    for i, cmd in enumerate(prep_cmds, 1):
        cmd_name = cmd[0] if isinstance(cmd, list) else cmd.split()[0]
        logging.info(f"--- Starting M population prep step [{i}/{total_steps}]: {cmd_name} ---")
        run_command(cmd, cmd_log_dir / f"prep_step_{i}.log", cwd=m_dir, env=env, module_load='warp/2.0.0dev39')

    refine_cmds = build_m_refine_command(cfg.m_refine_params)
    logging.info("--- Starting M refinement stage ---")
    total_steps = len(refine_cmds)
    for i, cmd in enumerate(refine_cmds, 1):
        cmd_name = cmd[0] if isinstance(cmd, list) else cmd.split()[0]
        logging.info(f"--- Starting M_refine step [{i}/{total_steps}]: {cmd_name} ---")
        run_command(cmd, cmd_log_dir / f"step_{i}.log", cwd=m_dir, env=env, module_load='warp/2.0.0dev39')
    
    logging.info("--- All M_refine steps completed successfully. ---")
    

def main():
    """Main function to initiate the appendix processing jobs."""
    parser = argparse.ArgumentParser(description="A stepwise handler for cryo-ET data processing.")
    parser.add_argument(
        '--stage',
        type=str,
        choices=['isonet', 'isonet2', 'cryolo', 'reconstruct', '3DTM', 'filter_3dtm', 'subtomo', 'm_refine', 'gapstop'],
        help="Which stage of the pipeline to run."
    )
    parser.add_argument('--input_list', type=str, default=None, help="Override input list file for 3DTM")
    args = parser.parse_args()

    if args.stage == '3DTM' and args.input_list:
            logging.info(f"Overriding input data list with: {args.input_list}")
            cfg.template_matching_params['input_data'] = args.input_list

    if cfg.dataset_name != Path.cwd().name:
        logging.warning(
            f"'{cfg.dataset_name}' in config does not match '{Path.cwd().name}'."
        )

    logs_dir = Path("logs")
    if not logs_dir.exists():
        logging.warning(f"Logs directory {logs_dir} does not exist. Did you run the main pipeline?")

    try:
        log_file_path = logs_dir / f"{args.stage}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, mode='w'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        logging.info(f"Main log file for this run is: {log_file_path.resolve()}")

        stage_map = {
            'reconstruct': reconstruction,
            'isonet': isonet,
            'isonet2': isonet2,
            'cryolo': cryolo,
            '3DTM': template_match_3D,
            'filter_3dtm': filter_3dtm,
            'subtomo': subtomo_extraction,
            'm_refine': m_refinement,
            'gapstop': template_match_gapstop
        }
        if args.stage in stage_map:
            stage_map[args.stage](log_file_path)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
