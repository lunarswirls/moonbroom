"""Reusable Lunar Prospector MAG and ER postprocessing"""

from .coordinates import (
    add_field_magnitude,
    compute_lonlat_from_sel,
    position_from_lonlat,
    sel_to_spherical,
    wrap_lon_180,
)
from .er import read_er, read_lper_txt
from .geometry import (
    classify_shadow_alt_sza,
    loaded_kernels,
    mean_radius_km,
    select_wake_from_nightside,
    sel_to_sse,
)
from .io import load_lp_csvs, read_pds4, write_csv
from .mag import load_pds4_lp, read_lpmag_l1b_tab, read_mag, read_mag_l2
from .pipeline import postprocess_mag, process_mag_directory
from .plotting import plot_timeseries
from .processing import (
    bayesian_trend_fit,
    build_design_matrix,
    compute_along_track_distance,
    detrend_track,
    detrend_tracks,
    ensure_datetime,
    filter_altitude_brms,
    filter_wake_file,
    segment_tracks_by_time,
    select_time_window,
)
from .raw import calibrate_er, calibrate_mag, read_raw_records
from .calibration import (
    assign_er_energies, calibrate_er_telemetry, calibrate_level0, calibrate_mag_telemetry,
    despin_mag, read_er_energy_bins,
)
from .timing import reconstruct_frame_times, reconstruct_product_times, reconstruct_spin_phase
from .archive_calibration import (
    apply_er_archive_response, apply_mag_archive_calibration, er_3d_reference,
    evaluate_mag_archive_calibration, fit_er_archive_response, fit_mag_archive_calibration,
    mag_archive_windows, mag_fixed_reference_windows, mag_sse_reference_to_scd, match_er_archive,
)
from .native_mag import (
    calibrate_mag_9hz, calibrate_mag_18hz, correct_mag_spin_offsets, fit_mag_spin_offsets, mag_calibration_diagnostics,
)
from .mag_state import annotate_mag_hardware_state, mag_hardware_calibration_checks
from .mission_calibration import (
    apply_er_mission_response, apply_pad_mission_response, prepare_archive_interval,
    prepare_mag_interval, select_mag_models,
)
from .pds_archive import archive_links, cache_pds_products, discover_lp_products
from .pad_calibration import (
    apply_pad_normalization, calibrate_er_pad, compare_pad_omni, er_omni_reference,
    fit_pad_normalization, pad_field_directions, pad_omnidirectional,
)
from .level0 import correlate_frame_times, decode_level0, read_merged_telemetry, read_sunpulse, write_level0_tables
from .mager import decode_mager_frames
from .er_telemetry import decode_er_accumulation
from .telemetry import (
    decode_digital_subcom,
    decode_mag_block,
    decompress_er_counts,
    er_angular_bins,
    er_energy_bins,
    er_frame_layout,
)

__version__ = "0.1.0"

from .archive_quality import (
    select_sunpulse_paths, read_sunpulse_products, read_outages, read_attitude,
    read_commands, maneuver_intervals, load_archive_quality,
    flag_archive_frames, flag_science_quality,
)
from .command_history import annotate_command_actions
from .timing import reconstruct_command_times

from .validation import (
    audit_mag_decoding, check_validation_split, compare_native_mag_reference,
    validate_mag_pseudocalibration, validate_er_pseudocalibration,
)
from .magnetotail import (
    shue_parameters, shue_tail_radius, native_variability_windows,
    spice_survey_geometry, rotate_scd_to_gse, survey_magnetotail,
    magnetotail_catalogue, variability_region_summary, variability_boundary_association,
)
