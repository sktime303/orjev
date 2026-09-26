"""Orjev-specific exceptions."""

from __future__ import annotations


class OrjevError(Exception):
    """Base class for recoverable Orjev errors."""


class MetadataError(OrjevError):
    """OpenRouter catalog or endpoint metadata could not be loaded."""


class JevError(OrjevError):
    """A Jev Decisions request or response was invalid."""


class PrivacyError(OrjevError):
    """Caller evidence could not be serialized within configured limits."""
