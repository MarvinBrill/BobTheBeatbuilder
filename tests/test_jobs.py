import threading
import time
from app import store
from app.jobs import Jobs


def test_cancelling_queued_job_does_not_run_operation():
    jobs = Jobs()
    gate = threading.Event()
    entered = threading.Event()
    ran = []
    cleaned = []

    def blocking(context):
        entered.set()
        assert gate.wait(5)

    jobs.submit("test", "blocker", blocking)
    assert entered.wait(5)
    second = jobs.submit(
        "test", "cancel me", lambda context: ran.append(True), cleanup=lambda: cleaned.append(True)
    )
    jobs.cancel(second["id"])
    gate.set()
    jobs.executor.shutdown(wait=True)
    assert store.get("job", second["id"])["status"] == "cancelled"
    assert not ran
    assert cleaned == [True]


def test_restart_marks_unfinished_jobs_interrupted():
    store.save("job", {"id": "stale-job", "status": "running"})
    jobs = Jobs()
    assert store.get("job", "stale-job")["status"] == "interrupted"
    jobs.executor.shutdown(wait=True)


def test_late_cancellation_does_not_hide_committed_result():
    jobs = Jobs()

    def operation(context):
        jobs.cancel(context.id)
        return {"output_id": "already-published"}

    job = jobs.submit("test", "commit wins", operation)
    jobs.executor.shutdown(wait=True)
    result = store.get("job", job["id"])
    assert result["status"] == "completed"
    assert result["result"]["output_id"] == "already-published"


def test_failed_artifact_directory_is_removed():
    from app.artifacts import artifact_directory
    from app.jobs import Cancelled
    import pytest

    with pytest.raises(Cancelled):
        with artifact_directory("models") as (_, directory):
            (directory / "partial.safetensors").write_bytes(b"partial")
            raise Cancelled()
    assert not directory.exists()
