from unittest.mock import MagicMock

import numpy as np

from holosoma.utils import video_utils
from holosoma.utils.wandb_utils import (
    configure_wandb_uploads,
    metrics_only_wandb_settings,
    wandb_file_uploads_enabled,
)


def test_wandb_upload_policy_defaults_to_metrics_only() -> None:
    configure_wandb_uploads(metrics_only=True)
    assert not wandb_file_uploads_enabled()


def test_wandb_upload_policy_allows_explicit_legacy_mode() -> None:
    configure_wandb_uploads(metrics_only=False)
    assert wandb_file_uploads_enabled()
    configure_wandb_uploads(metrics_only=True)


def test_metrics_only_settings_disable_auxiliary_run_files() -> None:
    wandb = MagicMock()

    metrics_only_wandb_settings(wandb)

    wandb.Settings.assert_called_once_with(
        console="off",
        disable_code=True,
        disable_git=True,
        save_code=False,
        x_disable_meta=True,
        x_save_requirements=False,
    )


def test_metrics_only_video_stays_local(tmp_path, monkeypatch) -> None:
    configure_wandb_uploads(metrics_only=True)
    wandb = MagicMock()
    monkeypatch.setattr(video_utils, "wandb", wandb)
    monkeypatch.setattr(video_utils, "_is_wandb_available", lambda: True)

    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    output = video_utils.create_video(
        frames,
        fps=10,
        save_dir=tmp_path,
        output_format="mp4",
        wandb_logging=True,
        episode_id=1,
    )

    assert output is not None
    assert output.is_file()
    wandb.Video.assert_not_called()
    wandb.log.assert_not_called()
