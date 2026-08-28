"""
NBEATSModel Explainer
----------------------

The `NBEATSExplainer` uses a trained :class:`NBEATSModel <darts.models.forecasting.nbeats.NBEATSModel>` (fitted
with the interpretable architecture, i.e. ``generic_architecture=False``) and exposes the trend and seasonality
components that the model's two stacks produce for a given forecast.

The trend and seasonality forecasts can be extracted using the :class:`NBEATSExplainabilityResult
<darts.explainability.explainability_result.NBEATSExplainabilityResult>` returned by
:func:`explain() <NBEATSExplainer.explain>`.
"""

from collections.abc import Sequence

from darts import TimeSeries
from darts.explainability import NBEATSExplainabilityResult
from darts.explainability.explainability import _ForecastingModelExplainer
from darts.logging import get_logger, raise_log
from darts.models import NBEATSModel
from darts.typing import TimeSeriesLike

logger = get_logger(__name__)


class NBEATSExplainer(_ForecastingModelExplainer):
    model: NBEATSModel

    def __init__(
        self,
        model: NBEATSModel,
        background_series: TimeSeriesLike | None = None,
        background_past_covariates: TimeSeriesLike | None = None,
    ):
        """
        Explainer class for the `NBEATSModel`.

        `NBEATSModel` can be trained with an "interpretable architecture" (``generic_architecture=False``),
        consisting of one trend stack and one seasonality stack. This explainer exposes the forecast produced by
        each of these two stacks separately, which decomposes the model's final forecast into a trend and a
        seasonality component.

        Parameters
        ----------
        model
            The fitted `NBEATSModel` to be explained. Must have been created with ``generic_architecture=False``
            and without a ``likelihood`` (only deterministic models are supported).
        background_series
            Optionally, a series or list of series to use as a default target series for the explanations.
            Optional if `model` was trained on a single target series. By default, it is the `series` used at
            fitting time. Mandatory if `model` was trained on multiple (sequence of) target series.
        background_past_covariates
            Optionally, a past covariates series or list of series to use as a default past covariates series
            for the explanations. The same requirements apply as for `background_series`.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.explainability import NBEATSExplainer
        >>> from darts.models import NBEATSModel
        >>> series = AirPassengersDataset().load()
        >>> model = NBEATSModel(
        >>>     input_chunk_length=24, output_chunk_length=12, generic_architecture=False
        >>> )
        >>> model.fit(series)
        >>> explainer = NBEATSExplainer(model)
        >>> results = explainer.explain()
        >>> trend = results.get_trend()
        >>> seasonality = results.get_seasonality()
        """
        if model.model is None:
            raise_log(
                ValueError(
                    f"The model must be fitted before instantiating a {self.__class__.__name__}."
                ),
            )
        if model.model.generic_architecture:
            raise_log(
                ValueError(
                    "`NBEATSExplainer` requires an `NBEATSModel` fitted with `generic_architecture=False` "
                    "(the interpretable architecture). With `generic_architecture=True`, the model's stacks "
                    "do not correspond to trend and seasonality components."
                ),
            )
        if model.model.nr_params != 1:
            raise_log(
                ValueError(
                    "`NBEATSExplainer` does not support probabilistic `NBEATSModel` (i.e. models fitted with a "
                    "`likelihood`); only deterministic models are currently supported."
                ),
            )
        super().__init__(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            requires_background=True,
            requires_covariates_encoding=False,
            check_component_names=False,
            test_stationarity=False,
        )

    def explain(
        self,
        foreground_series: TimeSeriesLike | None = None,
        foreground_past_covariates: TimeSeriesLike | None = None,
        foreground_future_covariates: TimeSeriesLike | None = None,
        horizons: Sequence[int] | None = None,
        target_components: Sequence[str] | None = None,
    ) -> NBEATSExplainabilityResult:
        """Returns the :class:`NBEATSExplainabilityResult
        <darts.explainability.explainability_result.NBEATSExplainabilityResult>` for all series in
        `foreground_series`. If `foreground_series` is `None`, will use the `background` input from
        `NBEATSExplainer` creation (either the `background` passed to creation, or the series stored in the
        `NBEATSModel` in case it was only trained on a single series).

        Parameters
        ----------
        foreground_series
            Optionally, one or a sequence of target `TimeSeries` to be explained. Can be multivariate.
            If not provided, the background `TimeSeries` will be explained instead.
        foreground_past_covariates
            Optionally, one or a sequence of past covariates `TimeSeries` if required by the forecasting model.
        foreground_future_covariates
            Not supported by `NBEATSExplainer` (`NBEATSModel` does not support future covariates).
        horizons
            This parameter is not used by the `NBEATSExplainer`.
        target_components
            This parameter is not used by the `NBEATSExplainer`.

        Returns
        -------
        NBEATSExplainabilityResult
            The explainability result containing the trend and seasonality forecast components.

        Examples
        --------
        >>> explainer = NBEATSExplainer(model)
        >>> explain_results = explainer.explain()
        >>> trend = explain_results.get_trend()
        >>> seasonality = explain_results.get_seasonality()
        """
        if (
            foreground_future_covariates is not None
            or horizons is not None
            or target_components is not None
        ):
            logger.warning(
                "`foreground_future_covariates`, `horizons`, and `target_components` are not supported by "
                "`NBEATSExplainer` and will be ignored."
            )
        (
            foreground_series,
            foreground_past_covariates,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self._process_foreground(
            foreground_series,
            foreground_past_covariates,
        )

        preds = self.model.predict(
            n=self.n,
            series=foreground_series,
            past_covariates=foreground_past_covariates,
        )

        # populated by `_NBEATSModule.forward()` during the `predict()` call above
        trend_forecast, seasonality_forecast = self.model.model._stacks_forecasts

        results = []
        for idx, (series, pred_series) in enumerate(zip(foreground_series, preds)):
            trend = TimeSeries(
                values=trend_forecast[idx, :, :, 0].detach().cpu().numpy(),
                times=pred_series.time_index,
                components=pred_series.columns,
            )
            seasonality = TimeSeries(
                values=seasonality_forecast[idx, :, :, 0].detach().cpu().numpy(),
                times=pred_series.time_index,
                components=pred_series.columns,
            )
            results.append({"trend": trend, "seasonality": seasonality})

        return NBEATSExplainabilityResult(
            explained_components=results[0] if len(results) == 1 else results
        )
