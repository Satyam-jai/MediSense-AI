"""
concurrency.py
--------------
Multithreading demo (Unit 2: Concurrency).

Several patients try to book the same appointment slot at the same time.
A threading.Lock ensures only one thread can successfully claim a given
slot — without it, race conditions could double-book the same slot.
"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field

from app.exceptions import SlotUnavailableError


@dataclass
class SlotBookingSystem:
    """Thread-safe booking of appointment slots."""
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _booked_slots: dict[str, str] = field(default_factory=dict)  # slot -> patient_id

    def book(self, slot: str, patient_id: str) -> None:
        # Simulate some processing time to make race conditions likely
        # without the lock (uncomment the sleep to observe raw races).
        with self._lock:
            if slot in self._booked_slots:
                raise SlotUnavailableError(slot)
            time.sleep(0.01)
            self._booked_slots[slot] = patient_id

    @property
    def bookings(self) -> dict[str, str]:
        return dict(self._booked_slots)


def simulate_concurrent_booking(slot: str, patient_ids: list[str]) -> dict:
    """
    Spins up one thread per patient, all racing to book the same slot.
    Returns a report of who succeeded/failed — useful as a demo/test.
    """
    system = SlotBookingSystem()
    results: dict[str, str] = {}
    results_lock = threading.Lock()

    def worker(pid: str):
        try:
            system.book(slot, pid)
            outcome = "BOOKED"
        except SlotUnavailableError:
            outcome = "REJECTED"
        with results_lock:
            results[pid] = outcome

    threads = [threading.Thread(target=worker, args=(pid,)) for pid in patient_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return {"slot": slot, "results": results, "winner": system.bookings.get(slot)}


if __name__ == "__main__":
    report = simulate_concurrent_booking(
        "2026-07-25 10:00",
        [f"P{i:03d}" for i in range(1, 11)],
    )
    print("Booking report:", report)
    winners = [pid for pid, outcome in report["results"].items() if outcome == "BOOKED"]
    assert len(winners) == 1, "Exactly one patient should win the slot"
    print(f"Exactly one patient ({winners[0]}) won the slot — lock worked correctly.")
