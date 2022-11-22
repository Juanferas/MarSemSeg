from dataclasses import dataclass
import typing as ty

@dataclass
class SegformerConfigFactory:

    def create(self) -> ty.Dict[str, ty.Any]:
        pass