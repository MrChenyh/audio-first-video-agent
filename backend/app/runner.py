from __future__ import annotations

from pathlib import Path

from .ai import AIClient
from .config import Settings
from .models import VideoAgentState
from .storage import JobStore
from .video import VideoProcessor
from .workflow import VideoUnderstandingWorkflow


class JobRunner:
    def __init__(self, settings: Settings, store: JobStore):
        self.settings = settings
        self.store = store

    def run_job(self, job_id: str, video_path: str, question: str) -> None:
        try:
            self.store.update_job(job_id, status="running", current_node="starting", progress=1)
            workflow = VideoUnderstandingWorkflow(
                self.settings,
                self.store,
                VideoProcessor(self.settings),
                AIClient(self.settings),
            )
            initial_state: VideoAgentState = {
                "job_id": job_id,
                "question": question,
                "video_path": str(Path(video_path)),
                "refinement_rounds": 0,
                "refinement_windows": [],
                "should_refine": False,
            }
            final_state = workflow.run(initial_state)
            result = final_state["result"]
            self.store.update_job(
                job_id,
                status="succeeded",
                current_node="complete",
                progress=100,
                result=result,
            )
            self.store.save_state(job_id, final_state)
        except Exception as exc:
            self.store.update_job(
                job_id,
                status="failed",
                current_node="failed",
                error=str(exc),
            )
