"""
exceptions.py
-------------
Custom exception hierarchy for MediSense AI (Unit 2: Exception Handling).

All domain-specific errors inherit from MediSenseError so calling code
can catch broadly (`except MediSenseError`) or narrowly
(`except DuplicatePatientError`) as needed.
"""


class MediSenseError(Exception):
    """Base class for all MediSense AI domain exceptions."""
    pass


class InvalidRecordError(MediSenseError):
    """Raised when a medical record fails validation (bad/missing fields)."""
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Invalid record field '{field}': {reason}")


class DuplicatePatientError(MediSenseError):
    """Raised when attempting to register a patient with an existing ID."""
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        super().__init__(f"Patient with ID '{patient_id}' already exists")


class PatientNotFoundError(MediSenseError):
    """Raised when a lookup is performed for a patient that doesn't exist."""
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        super().__init__(f"No patient found with ID '{patient_id}'")


class SlotUnavailableError(MediSenseError):
    """Raised when an appointment slot is already booked (used by Unit 2's
    concurrency demo to show thread-safe booking)."""
    def __init__(self, slot: str):
        self.slot = slot
        super().__init__(f"Appointment slot '{slot}' is already booked")


class FileProcessingError(MediSenseError):
    """Raised when reading/writing patient data files (CSV/pickle) fails."""
    pass
