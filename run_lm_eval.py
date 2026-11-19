import sys
import fla  # registers the fla models (DeltaFugue, GatedDeltaFugue, ...) with AutoConfig
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    sys.exit(cli_evaluate())
