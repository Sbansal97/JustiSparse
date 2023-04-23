import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DebiasArguments:
    "Arguments pertaining to sparse fine-tuning configuration."""
    
    debias_configuration: Optional[str] = field(
        default='none', metadata={"help": "which configuration should be used. choose between [none, before, after]"}
    )

    diffs_path: str = field(
        default=None, metadata={"help": "Optional path to weight diffs"}
    )

    def __post_init__(self):

        assert self.debias_configuration in ['none', 'before', 'after']
        if self.debias_configuration != 'none':
            assert os.path.isfile(self.diffs_path)