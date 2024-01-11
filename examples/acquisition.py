import fire
from ax.modelbridge import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.ax_client import AxClient, ObjectiveProperties
from botorch.acquisition import qMaxValueEntropy
from tqdm import tqdm

from bocoel.core.optim.ax.acquisition import MaxEntropy

K = "K"


def main(acqf: str):
    if acqf == "qMaxValueEntropy":
        botorch_acqf_class = qMaxValueEntropy
    elif acqf == "MaxEntropy":
        botorch_acqf_class = MaxEntropy

    cli = AxClient(
        GenerationStrategy(
            [
                GenerationStep(Models.SOBOL, num_trials=5),
                GenerationStep(
                    Models.BOTORCH_MODULAR,
                    num_trials=-1,
                    model_kwargs={
                        "torch_device": "cpu",
                        "botorch_acqf_class": botorch_acqf_class,
                    },
                ),
            ]
        )
    )

    cli.create_experiment(
        parameters=[
            {"name": f"x{i}", "type": "range", "bounds": [-1, 1], "value_type": "float"}
            for i in range(32)
        ],
        objectives={K: ObjectiveProperties(minimize=True)},
    )
    for i in tqdm(range(10)):
        param, idx = cli.get_next_trial()
        cli.complete_trial(idx, raw_data={K: 0})


if __name__ == "__main__":
    fire.Fire(main)
