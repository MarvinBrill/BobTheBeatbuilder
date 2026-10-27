import threading
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from . import store


class Cancelled(Exception):
    pass


class Jobs:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bob-worker")
        self.events = {}
        self.uploads = 0
        self.lock = threading.RLock()
        for job in store.all_records("job"):
            if job["status"] in ("queued", "running"):
                store.save(
                    "job",
                    {
                        **job,
                        "status": "interrupted",
                        "message": "App restarted; please start a new job.",
                    },
                )

    def submit(self, kind, name, operation, cleanup=None):
        # Serialise job admission with deletion and resource validation in the API.
        with self.lock:
            return self._submit(kind, name, operation, cleanup)

    def _submit(self, kind, name, operation, cleanup):
        identifier = uuid4().hex
        self.events[identifier] = threading.Event()
        job = store.save(
            "job",
            {
                "id": identifier,
                "kind": kind,
                "name": name,
                "status": "queued",
                "progress": 0,
                "message": "Waiting for worker",
                "created_at": store.now(),
                "metrics": [],
            },
        )
        self.executor.submit(self.run, identifier, operation, cleanup)
        return job

    def update(self, identifier, **changes):
        with self.lock:
            return store.save("job", {**store.get("job", identifier), **changes})

    def check(self, identifier):
        if self.events[identifier].is_set():
            raise Cancelled()

    def run(self, identifier, operation, cleanup=None):
        try:
            self.check(identifier)
            self.update(identifier, status="running", message="Starting")
            result = operation(Context(self, identifier))
            # Operations check cancellation before publishing. A committed result stays completed.
            self.update(
                identifier, status="completed", progress=1, message="Completed", result=result
            )
        except Cancelled:
            self.update(identifier, status="cancelled", message="Cancelled")
        except Exception as error:
            traceback.print_exc()
            self.update(identifier, status="failed", message=str(error))
        finally:
            try:
                if cleanup:
                    try:
                        cleanup()
                    except OSError:
                        traceback.print_exc()
                        self.update(
                            identifier,
                            cleanup_warning="Temporärer Upload konnte nicht entfernt werden. Bitte in der Speicherverwaltung prüfen.",
                        )
            finally:
                with self.lock:
                    try:
                        from .storage_files import prune_job_history, trim_analysis_cache

                        trim_analysis_cache()
                        prune_job_history()
                    except Exception:
                        logging.exception(
                            "Analysis cache cleanup failed; inspect Storage in the UI"
                        )
                        self.update(
                            identifier,
                            cleanup_warning="Zwischendateien konnten nicht vollständig bereinigt werden. Bitte in der Speicherverwaltung prüfen.",
                        )
                    finally:
                        self.events.pop(identifier, None)

    def cancel(self, identifier):
        event = self.events.get(identifier)
        if event:
            event.set()
        return store.get("job", identifier)


class Context:
    def __init__(self, jobs, identifier):
        self.jobs, self.id = jobs, identifier

    def check(self):
        self.jobs.check(self.id)

    def report(self, progress, message, **extra):
        self.check()
        self.jobs.update(self.id, progress=progress, message=message, **extra)
