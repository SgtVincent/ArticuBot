# Import all functions from the modular utils for backward compatibility
from .defaults import *
from .env_utils import *
from .geometry_utils import *
from .handle_utiils import *
from .object_utils import *

# Add deprecation warning for old utils.py usage
# import warnings
# warnings.warn(
#     "Importing from utils.py is deprecated. "
#     "Please import directly from the specific utils modules: "
#     "defaults, env_utils, geometry_utils, handle_utiils, object_utils",
#     DeprecationWarning,
#     stacklevel=2
# )