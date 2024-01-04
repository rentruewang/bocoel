from ax.modelbridge import ModelBridge, cross_validation
from ax.plot import contour, diagnostic, scatter, slice
from ax.service.ax_client import AxClient
from ax.utils.notebook import plotting


def gen_strat(ax_client: AxClient) -> ModelBridge:
    model = ax_client.generation_strategy.model
    assert model is not None
    return model


def render_static(
    ax_client: AxClient, param_x: str, param_y: str, metric_name: str, **_
) -> None:
    plotting.render(
        ax_client.get_contour_plot(
            param_x=param_x, param_y=param_y, metric_name=metric_name
        )
    )


def render_interactive(ax_client: AxClient, metric_name: str, **_) -> None:
    plotting.render(
        contour.interact_contour(model=gen_strat(ax_client), metric_name=metric_name)
    )


def render_tradeoff(ax_client: AxClient, metric_name: str, **_) -> None:
    plotting.render(
        scatter.plot_objective_vs_constraints(
            model=gen_strat(ax_client), objective=metric_name, rel=False
        )
    )


def render_cross_validate(ax_client: AxClient, **_) -> None:
    plotting.render(
        diagnostic.interact_cross_validation(
            cv_results=cross_validation.cross_validate(gen_strat(ax_client))
        )
    )


def render_slice(ax_client: AxClient, param_name: str, metric_name: str, **_) -> None:
    plotting.render(
        slice.plot_slice(
            model=gen_strat(ax_client),
            param_name=param_name,
            metric_name=metric_name,
        )
    )


def render_tile(ax_client: AxClient, **_) -> None:
    plotting.render(scatter.interact_fitted(model=gen_strat(ax_client), rel=False))
