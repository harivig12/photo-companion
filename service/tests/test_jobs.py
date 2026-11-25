from app.jobs import JobStore


def test_queue_order_claim_and_finish(db):
    store = JobStore(db)
    a = store.enqueue("collage", "A", {"asset_ids": ["1", "2"]})
    b = store.enqueue("cinematic", "B", {"asset_ids": ["3"]}, origin="suggestion")
    assert [j.id for j in store.queued()] == [a.id, b.id]

    claimed = store.claim_next()
    assert claimed.id == a.id and claimed.status == "running"
    store.progress(a.id, "Render", 150)
    assert store.get(a.id).progress == 100  # clamped

    store.finish(a.id, "done", result={"outputs": []})
    done = store.get(a.id)
    assert done.status == "done" and done.finished_at
    assert store.counts()["done_today"] == 1
    assert store.claim_next().id == b.id
    assert store.claim_next() is None


def test_move_up_swaps_with_job_ahead(db):
    store = JobStore(db)
    a, b, c = (store.enqueue("collage", t, {}) for t in "ABC")
    assert store.move_up(c.id)
    assert [j.title for j in store.queued()] == ["A", "C", "B"]
    assert not store.move_up(a.id)  # already first


def test_cancel_queued_and_running(db):
    store = JobStore(db)
    a = store.enqueue("collage", "A", {})
    b = store.enqueue("collage", "B", {})
    store.claim_next()  # A running
    assert store.cancel(b.id)
    assert store.get(b.id).status == "cancelled"
    assert store.cancel(a.id)
    assert store.get(a.id).status == "running"  # stops at its next step
    assert store.is_cancel_requested(a.id)


def test_recover_puts_running_back_at_front(db):
    store = JobStore(db)
    a = store.enqueue("collage", "A", {})
    store.enqueue("collage", "B", {})
    store.claim_next()
    assert store.recover_interrupted() == 1
    assert store.queued()[0].id == a.id


def test_retry_skipped_forces_run(db):
    store = JobStore(db)
    a = store.enqueue("cinematic", "flat", {"asset_ids": ["x"]})
    store.claim_next()
    store.finish(a.id, "skipped", "Flat scene")
    again = store.retry(a.id)
    assert again.params == {"asset_ids": ["x"], "force": True}
    assert store.retry(again.id) is None  # queued jobs can't be retried


def test_rejects_unknown_kind(db):
    import pytest

    with pytest.raises(ValueError):
        JobStore(db).enqueue("animation", "x", {})


def test_counts_by_kind_today(db):
    store = JobStore(db)
    a = store.enqueue("collage", "A", {})
    store.enqueue("collage", "B", {})
    store.claim_next()
    store.finish(a.id, "done")
    assert store.counts_by_kind_today()["collage"] == {"done": 1, "planned": 2}
    assert store.counts_by_kind_today()["highlight"] == {"done": 0, "planned": 0}
