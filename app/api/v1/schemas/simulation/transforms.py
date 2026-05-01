"""Parameter transform schemas: balcan / scale / override (discriminated union)."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class _BaseTransform(BaseModel):
    """Common fields for all parameter transforms."""

    target_parameter: str = Field(
        ...,
        description="Name of the model parameter to transform. Must already be defined in `model.parameters`.",
    )


class BalcanTransform(_BaseTransform):
    """Balcan-style sinusoidal seasonality applied across the simulation timeline (Balcan D et al. J. Comput. Sci. 2010; https://doi.org/10.1016/j.jocs.2010.07.002).

    The transform is multiplicative: at each step, the existing parameter value
    is multiplied by a factor in `[min_value/max_value, 1]` so that the rate
    peaks at `max_value` on `max_date` and troughs at `min_value` on `min_date`
    (or half a period away if `min_date` is omitted).
    """

    method: Literal["balcan"]
    max_date: str = Field(..., description="Date of peak value, `YYYY-MM-DD`.")
    max_value: float = Field(..., description="Maximum parameter value (at `max_date`).")
    min_value: float = Field(..., description="Minimum parameter value (at `min_date` or half a period away).")
    min_date: str | None = Field(
        default=None,
        description=(
            "Date of trough, `YYYY-MM-DD`. If set, period = `2 * |min_date - max_date|`. "
            "If omitted, period defaults to 365 days."
        ),
    )


class ScaleTransform(_BaseTransform):
    """Multiplicative scaling applied during a date window.

    Outside `[start_date, end_date]` the multiplier is 1.0; inside it is
    `factor`. Composes multiplicatively with other transforms on the same
    parameter in the order listed.
    """

    method: Literal["scale"]
    start_date: str = Field(..., description="Window start, `YYYY-MM-DD`.")
    end_date: str = Field(..., description="Window end, `YYYY-MM-DD`. Must be `>= start_date`.")
    factor: float = Field(..., description="Multiplicative factor applied within `[start_date, end_date]`.")

    @model_validator(mode="after")
    def _validate_window(self) -> "ScaleTransform":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")
        return self


class OverrideTransform(_BaseTransform):
    """Absolute value override during a date window.

    Replaces the parameter wholesale during `[start_date, end_date]`. Stored in
    `model.overrides` separately from `model.parameters`, so an override always
    wins for its window regardless of where it appears in the transform list.
    """

    method: Literal["override"]
    start_date: str = Field(..., description="Window start, `YYYY-MM-DD`.")
    end_date: str = Field(..., description="Window end, `YYYY-MM-DD`. Must be `>= start_date`.")
    value: float | list[float] = Field(
        ...,
        description=(
            "Absolute replacement value during the window. Scalar for uniform, "
            "or a list of one value per age group (length must match the resolved population)."
        ),
    )

    @model_validator(mode="after")
    def _validate_window(self) -> "OverrideTransform":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")
        return self


ParameterTransformConfig = Annotated[
    BalcanTransform | ScaleTransform | OverrideTransform,
    Field(discriminator="method"),
]
