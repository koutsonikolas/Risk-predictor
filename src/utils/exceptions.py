"""Project-specific exceptions."""


class EarthRiskError(Exception):
    """Base error for project failures."""


class ConfigurationError(EarthRiskError):
    """Raised when configuration is missing or invalid."""


class MissingInputError(EarthRiskError):
    """Raised when a required local input file is missing."""


class DataValidationError(EarthRiskError):
    """Raised when input data cannot be validated."""


class EarthEngineSetupError(EarthRiskError):
    """Raised when Google Earth Engine is not ready for use."""
