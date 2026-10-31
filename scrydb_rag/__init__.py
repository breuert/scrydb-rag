"""Retrieval-augmented generation over scrydb indices with the models of an OpenAI-compatible API."""

from thki import APIError, ThkiError


class RagError(ThkiError):
    """Error that is reported to the user without a traceback."""


class ContextError(APIError):
    """Reply to a prompt that did not fit into the context of the model, which therefore lost a part of it."""
