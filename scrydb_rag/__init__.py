"""Retrieval-augmented generation over scrydb indices with the models of the KI:connect.nrw platform."""

from thki import ThkiError


class RagError(ThkiError):
    """Error that is reported to the user without a traceback."""
