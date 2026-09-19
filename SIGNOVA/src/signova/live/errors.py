"""
Error Hierarchy for SIGNOVA Live Camera Runtime.
"""


class LiveRuntimeError(Exception):
    """Base exception for all live runtime errors."""
    pass


class CameraError(LiveRuntimeError):
    """Raised when camera initialization, capture, or stream fails."""
    pass


class TrackingError(LiveRuntimeError):
    """Raised when MediaPipe landmark tracking fails or returns invalid topology."""
    pass


class ModelIncompatibleError(LiveRuntimeError):
    """Raised when live feature specification does not match model expectations."""
    pass


class ModelUnauthorizedError(LiveRuntimeError):
    """Raised when an attempt is made to load an unauthorized or synthetic model as real."""
    pass


class SessionError(LiveRuntimeError):
    """Raised when live session state management encounters an inconsistency."""
    pass
