from mmcv import Config
from .iconfig import IConfigAdapter


class PSPNetAdapter(IConfigAdapter):

    def adapt(self, base_config: Config) -> Config:
        pass


class SegformerAdapter(IConfigAdapter):

    def create(self, base_config: Config) -> Config:
        pass