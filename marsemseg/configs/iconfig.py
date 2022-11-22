from abc import ABC, abstractmethod
from mmcv import Config


class IConfigAdapter(ABC):

    @abstractmethod
    def adapt(self, base_config: Config) -> Config:
        raise NotImplementedError("Subclasses should implement this method (create)!")
