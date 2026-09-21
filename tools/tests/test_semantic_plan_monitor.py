import json

import monitor_remaining37_semantic_plans as monitor


def test_incomplete_plan_never_emits_completion(tmp_path):
    rows = [{"dataset": "omomo", "task": "table", "status": "pending", "output": "missing.json"}]
    monitor.publish(tmp_path, rows, "waiting_for_quota", {}, 100, last_error="insufficient_quota")
    assert not (tmp_path / "COMPLETE.md").exists()
    assert json.loads((tmp_path / "monitor_status.json").read_text())["validated"] == 0


def test_all_validated_emits_completion_with_visual_review_pending(tmp_path):
    rows = [{"dataset": "omomo", "task": str(i), "status": "validated", "output": f"{i}.json"} for i in range(37)]
    assert monitor.publish(tmp_path, rows, "checking", {}, 100) == 37
    assert (tmp_path / "COMPLETE.md").is_file()
    state = json.loads((tmp_path / "monitor_status.json").read_text())
    assert state["state"] == "complete_pending_visual_review"
    assert state["new_training_started"] is False


def test_corrupt_plan_is_rejected(tmp_path):
    directory = tmp_path / "omomo/table"
    directory.mkdir(parents=True)
    (directory / "semantic_plan.json").write_text("{invalid")
    (directory / "semantic_plan.event_plan.json").write_text("{}")
    assert monitor.validate_output(tmp_path, "omomo", "table")["status"] == "invalid"


def test_restart_restores_failed_tasks_and_retry_budget(tmp_path):
    rows = [{"dataset": "omomo", "task": "table", "status": "pending", "output": "missing.json"}]
    failure = tmp_path / "monitor_attempts/round_001/omomo/table/result.json"
    failure.parent.mkdir(parents=True)
    failure.write_text(
        json.dumps(
            {
                "task": "table",
                "status": "failed",
                "error": "VLM failed to produce a valid dynamic action plan: duplicate frames",
            }
        )
    )
    attempts, failures, round_id = monitor.restore_progress(tmp_path, rows)
    assert attempts == {"table": 1} and failures == {"table": 1} and round_id == 1
    assert rows[0]["status"] == "failed_validation"
    monitor.publish(tmp_path, rows, "checking", attempts, 100)
    report = (tmp_path / "MONITOR.md").read_text()
    assert "failed_validation" in report and "duplicate frames" in report
    assert not (tmp_path / "COMPLETE.md").exists()


def test_jobs_skipped_after_quota_are_reported_as_quota_blocked():
    assert monitor.failure_status("provider quota exhausted earlier in batch") == "quota_blocked"
    assert monitor.failure_status("VLMQuotaError: insufficient_quota") == "quota_blocked"
    assert monitor.failure_status("VLM failed to produce a valid localized plan") == "failed_validation"
