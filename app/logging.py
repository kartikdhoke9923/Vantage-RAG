"""Per-request correlation ID support for the FastAPI app."""
import contextvars

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="unknown")


def set_request_id(request_id: str) -> None:
    """Bind a correlation id to the current async context."""
    _request_id.set(request_id)


def get_request_id() -> str:
    """Return the correlation id bound to the current context."""
    return _request_id.get()