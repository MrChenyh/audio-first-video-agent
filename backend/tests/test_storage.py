from app.storage import JobStore


def test_state_roundtrip(tmp_path):
    store = JobStore(tmp_path)
    store.init()
    state = {"job_id": "job1", "question": "q", "nested": {"value": 1}}

    store.save_state("job1", state)

    assert store.load_state("job1") == state
