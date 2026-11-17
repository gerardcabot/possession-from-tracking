import numpy as np

from possession.features import Features, central_velocity

PITCH = (100.0, 60.0)
FPS = 25.0


def make_features(n_frames: int, ball: np.ndarray | None = None,
                  nearest: np.ndarray | None = None, mean_speed: float | np.ndarray = 3.0,
                  **overrides: object) -> Features:
    ball = np.full((n_frames, 2), 50.0) if ball is None else ball
    velocity = central_velocity(ball, FPS, 1)
    nearest = np.full((n_frames, 2), 10.0) if nearest is None else nearest
    values = {
        "ball": ball,
        "ball_present": ~np.isnan(ball[:, 0]),
        "ball_velocity": velocity,
        "ball_speed": np.linalg.norm(velocity, axis=1),
        "nearest": nearest,
        "second_nearest": nearest + 5.0,
        "centroid": np.full((n_frames, 2, 2), 50.0),
        "centroid_velocity": np.zeros((n_frames, 2, 2)),
        "defensive_line": np.full((n_frames, 2), 30.0),
        "team_speed": np.full((n_frames, 2), 3.0),
        "mean_speed": np.broadcast_to(np.asarray(mean_speed, dtype=float), n_frames).copy(),
        "attack_sign": np.ones(n_frames, dtype=np.int8),
        "tracked": np.full((n_frames, 2), 11),
        "beyond_halfway": np.full((n_frames, 2), 5),
        "centre_distance": np.full((n_frames, 2), 20.0),
        "pitch": PITCH,
        "fps": FPS,
    }
    values.update(overrides)
    return Features(**values)


def still_then_kicked(n_frames: int, kick: int, at: tuple[float, float],
                      speed_mps: float = 10.0) -> np.ndarray:
    ball = np.tile(np.asarray(at, dtype=float), (n_frames, 1))
    steps = np.arange(n_frames - kick) * speed_mps / FPS
    ball[kick:, 0] += steps
    return ball
