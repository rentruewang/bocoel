from bocoel import DataFrameStorage, DatasetsStorage, Storage


def storage(name: str) -> type[Storage]:
    match name:
        case "datasets":
            return DatasetsStorage
        case "dataframe" | "df":
            return DataFrameStorage
        case _:
            raise ValueError(f"Unknown storage: {name}")
