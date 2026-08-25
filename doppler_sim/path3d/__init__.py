"""3D path whiteboard tab."""

from doppler_sim.path3d.synthesis import (
    resample_path_constant_speed,
    solve_retarded_time_path3d,
    synthesize_path3d_audio,
)

__all__ = [
    "resample_path_constant_speed",
    "solve_retarded_time_path3d",
    "synthesize_path3d_audio",
]
