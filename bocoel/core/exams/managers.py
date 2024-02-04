import datetime as dt
import hashlib
from collections import OrderedDict
from collections.abc import Generator, Mapping
from pathlib import Path

import alive_progress as ap
import pandas as pd
from pandas import DataFrame

from bocoel.core.optim import Optimizer
from bocoel.corpora import Corpus, Embedder
from bocoel.models import Adaptor, ClassifierModel, GenerativeModel

from .examinators import Examinator

DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
META_DATA = "metadata.json"

TIME = "time"
MD5 = "md5"

INDEX = "index"
STORAGE = "storage"
EMBEDDER = "embedder"
OPTIMIZER = "optimizer"
MODEL = "model"
ADAPTOR = "adaptor"


class Manager:
    def __init__(
        self, path: str | Path, batch_size: int = 64, cuda: bool = False
    ) -> None:
        self.path = Path(path)
        if self.path.exists() and not self.path.is_dir():
            raise ValueError(f"{path} is not a directory")
        self.path.mkdir(parents=True, exist_ok=True)

        self._start = self.current()
        self.examinator = Examinator.presets(batch_size=batch_size, cuda=cuda)

    def run(
        self, optimizer: Optimizer, corpus: Corpus, steps: int | None = None
    ) -> DataFrame:
        """
        Runs the optimizer until the end.

        Parameters
        ----------

        `save_path: str | Path`
        The path to save the results to, if given.

        Returns
        -------

        The final state of the optimizer.
        Keys are the indices of the queries,
        and values are the corresponding scores.
        """

        results: OrderedDict[int, float] = OrderedDict()
        for res in self._launch(optimizer=optimizer, steps=steps):
            results.update(res)

        scores = self.examinator.examine(index=corpus.index, results=results)
        return scores

    def save(
        self,
        scores: DataFrame,
        optimizer: Optimizer,
        corpus: Corpus,
        model: GenerativeModel | ClassifierModel,
        adaptor: Adaptor,
        embedder: Embedder,
    ) -> None:
        md5, scores = self.with_identifier_cols(
            scores, optimizer, corpus, model, adaptor, embedder
        )

        scores.to_csv(self.path / f"{md5}.csv", index=False)

    def _launch(
        self, optimizer: Optimizer, steps: int | None = None
    ) -> Generator[Mapping[int, float], None, None]:
        "Launches the optimizer as a generator."

        steps_criterion = float(steps) if steps is not None else float("inf")

        with ap.alive_bar(total=steps, title="optimizing") as bar:
            while steps_criterion > 0:
                bar()
                steps_criterion -= 1

                # Raises StopIteration (converted to RuntimError per PEP 479) if done.
                try:
                    results = optimizer.step()
                except StopIteration:
                    break

                yield results

    def with_identifier_cols(
        self,
        df: DataFrame,
        optimizer: Optimizer,
        corpus: Corpus,
        model: GenerativeModel | ClassifierModel,
        adaptor: Adaptor,
        embedder: Embedder,
    ) -> tuple[str, DataFrame]:
        df = df.copy()

        md5 = self.md5(optimizer, corpus, model, adaptor, embedder, self._start)

        df[OPTIMIZER] = [str(optimizer)] * len(df)
        df[MODEL] = [str(model)] * len(df)
        df[ADAPTOR] = [str(adaptor)] * len(df)
        df[INDEX] = [str(corpus.index.index)] * len(df)
        df[STORAGE] = [str(corpus.storage)] * len(df)
        df[EMBEDDER] = [str(embedder)] * len(df)
        df[TIME] = [self._start] * len(df)
        df[MD5] = [md5] * len(df)

        return md5, df

    @staticmethod
    def load(path: str | Path) -> DataFrame:
        path = Path(path)

        dfs: list[DataFrame] = []

        for csv in path.iterdir():
            if csv.suffix != ".csv":
                raise ValueError(f"{csv} is not a csv file")

            df = pd.read_csv(csv)
            dfs.append(df)

        if not dfs:
            raise ValueError(f"No csv files found in {path}")

        return pd.concat(dfs)

    @staticmethod
    def md5(
        optimizer: Optimizer,
        corpus: Corpus,
        model: GenerativeModel | ClassifierModel,
        adaptor: Adaptor,
        embedder: Embedder,
        time: str,
    ) -> str:
        data = [
            optimizer,
            embedder,
            corpus.index.index,
            corpus.storage,
            model,
            adaptor,
            time,
        ]

        return hashlib.md5(
            str.encode(" ".join([str(item) for item in data]))
        ).hexdigest()

    @staticmethod
    def current() -> str:
        return dt.datetime.now().strftime(DATETIME_FORMAT)
