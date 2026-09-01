from types import SimpleNamespace

from holosoma.utils.experiment_paths import get_eval_log_dir


def test_eval_sessions_are_stored_under_exp() -> None:
    logger = SimpleNamespace(base_dir="legacy_logs", project=None, name=None)
    training = SimpleNamespace(project="WholeBodyTracking", name="b4_omni")

    path = get_eval_log_dir(logger, training, "20260826_130000")

    assert path.as_posix() == (
        "exp/eval/sessions/WholeBodyTracking/20260826_130000-b4_omni-eval"
    )
