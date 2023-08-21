from typing import List
from dataclasses import dataclass, field
from typing import List, Optional, Union

from datasets.features import Features, Value

@dataclass
class MetricInfo:

    description: str
    citation: str
    features: Union[Features, List[Features]]
    upper_bound: int
    lower_bound: int
    inputs_description: str = field(default_factory=str)
    homepage: str = field(default_factory=str)
    license: str = field(default_factory=str)
    codebase_urls: List[str] = field(default_factory=list)
    reference_urls: List[str] = field(default_factory=list)
    streamable: bool = False
    format: Optional[str] = None
    module_type: str = "metric"

    # Set later by the builder
    module_name: Optional[str] = None
    config_name: Optional[str] = None
    experiment_id: Optional[str] = None

    def __post_init__(self):
        if self.format is not None:
            for key, value in self.features.items():
                if not isinstance(value, Value):
                    raise ValueError(
                        f"When using 'numpy' format, all features should be a `datasets.Value` feature. "
                        f"Here {key} is an instance of {value.__class__.__name__}"
                    )