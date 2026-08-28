import numpy as np
import pytest

from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import timeseries_generation as tg

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
from darts.explainability import NBEATSExplainabilityResult, NBEATSExplainer
from darts.models import NBEATSModel
from darts.utils.likelihood_models.torch import GaussianLikelihood


class TestNBEATSExplainer:
    freq = "MS"
    icl = 24
    ocl = 12
    series = tg.linear_timeseries(length=icl + ocl + 10, freq=freq)
    series_mv = series.stack(tg.sine_timeseries(length=icl + ocl + 10, freq=freq))
    pc = tg.constant_timeseries(length=icl + ocl + 10, freq=freq)

    def helper_fit_model(self, **model_kwargs) -> NBEATSModel:
        model = NBEATSModel(
            input_chunk_length=self.icl,
            output_chunk_length=self.ocl,
            generic_architecture=False,
            n_epochs=1,
            random_state=42,
            **tfm_kwargs,
            **model_kwargs,
        )
        model.fit(self.series)
        return model

    def test_explain_reconstructs_prediction_univariate(self):
        """trend + seasonality forecasts must sum up to exactly the model's prediction."""
        model = self.helper_fit_model()
        explainer = NBEATSExplainer(model)
        result = explainer.explain()
        assert isinstance(result, NBEATSExplainabilityResult)

        trend = result.get_trend()
        seasonality = result.get_seasonality()
        pred = model.predict(n=self.ocl)

        assert trend.time_index.equals(pred.time_index)
        assert seasonality.time_index.equals(pred.time_index)
        np.testing.assert_allclose(
            trend.values() + seasonality.values(), pred.values(), atol=1e-4
        )

    def test_explain_multivariate_series(self):
        model = NBEATSModel(
            input_chunk_length=self.icl,
            output_chunk_length=self.ocl,
            generic_architecture=False,
            n_epochs=1,
            random_state=42,
            **tfm_kwargs,
        )
        model.fit(self.series_mv)
        explainer = NBEATSExplainer(model)
        result = explainer.explain()

        pred = model.predict(n=self.ocl)
        trend = result.get_trend()
        seasonality = result.get_seasonality()
        assert trend.n_components == pred.n_components == 2
        np.testing.assert_allclose(
            trend.values() + seasonality.values(), pred.values(), atol=1e-4
        )

    def test_explain_multiple_series_returns_list(self):
        series2 = tg.linear_timeseries(
            start_value=1, end_value=0, length=self.icl + self.ocl + 10, freq=self.freq
        )
        model = NBEATSModel(
            input_chunk_length=self.icl,
            output_chunk_length=self.ocl,
            generic_architecture=False,
            n_epochs=1,
            random_state=42,
            **tfm_kwargs,
        )
        model.fit([self.series, series2])
        explainer = NBEATSExplainer(model, background_series=[self.series, series2])
        result = explainer.explain()

        trends = result.get_trend()
        seasonalities = result.get_seasonality()
        assert isinstance(trends, list) and len(trends) == 2
        assert isinstance(seasonalities, list) and len(seasonalities) == 2

    def test_explain_with_past_covariates(self):
        model = NBEATSModel(
            input_chunk_length=self.icl,
            output_chunk_length=self.ocl,
            generic_architecture=False,
            n_epochs=1,
            random_state=42,
            **tfm_kwargs,
        )
        model.fit(self.series, past_covariates=self.pc)
        explainer = NBEATSExplainer(model)
        result = explainer.explain()
        pred = model.predict(n=self.ocl, series=self.series, past_covariates=self.pc)
        np.testing.assert_allclose(
            result.get_trend().values() + result.get_seasonality().values(),
            pred.values(),
            atol=1e-4,
        )

    def test_generic_architecture_raises(self):
        model = NBEATSModel(
            input_chunk_length=self.icl,
            output_chunk_length=self.ocl,
            generic_architecture=True,
            n_epochs=1,
            random_state=42,
            **tfm_kwargs,
        )
        model.fit(self.series)
        with pytest.raises(ValueError, match="generic_architecture=False"):
            NBEATSExplainer(model)

    def test_unfitted_model_raises(self):
        model = NBEATSModel(
            input_chunk_length=self.icl,
            output_chunk_length=self.ocl,
            generic_architecture=False,
        )
        with pytest.raises(ValueError, match="must be fitted"):
            NBEATSExplainer(model)

    def test_probabilistic_model_raises(self):
        model = NBEATSModel(
            input_chunk_length=self.icl,
            output_chunk_length=self.ocl,
            generic_architecture=False,
            likelihood=GaussianLikelihood(),
            n_epochs=1,
            random_state=42,
            **tfm_kwargs,
        )
        model.fit(self.series)
        with pytest.raises(ValueError, match="probabilistic"):
            NBEATSExplainer(model)
