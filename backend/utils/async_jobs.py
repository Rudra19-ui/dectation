import threading
import uuid

class JobStatus:
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class _JobRecord:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = JobStatus.PROCESSING
        self.result = None
        self.error = None

class _JobQueue:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = _JobRecord(job_id)
        return job_id

    def set_result(self, job_id: str, result):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.result = result
                job.status = JobStatus.COMPLETED

    def set_error(self, job_id: str, error: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.error = error
                job.status = JobStatus.FAILED

    def get_job_status(self, job_id: str):
        with self._lock:
            return self._jobs.get(job_id)

job_queue = _JobQueue()

def _run_job(job_id: str, image_bytes: bytes, process_fn):
    try:
        result = process_fn(image_bytes)
        job_queue.set_result(job_id, result)
    except Exception as e:
        job_queue.set_error(job_id, str(e))

def enqueue_large_image_job(image_bytes: bytes, process_fn, background_tasks=None) -> str:
    job_id = job_queue.create_job()
    if background_tasks is not None:
        background_tasks.add_task(_run_job, job_id, image_bytes, process_fn)
    else:
        t = threading.Thread(target=_run_job, args=(job_id, image_bytes, process_fn), daemon=True)
        t.start()
    return job_id