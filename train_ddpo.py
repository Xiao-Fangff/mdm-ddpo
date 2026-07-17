from __future__ import annotations

import json
import logging

from mdm_ddpo.config import parse_config
from mdm_ddpo.trainer import DDPOTrainer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = parse_config()
    trainer = DDPOTrainer(config)
    if config.preflight:
        print(json.dumps(trainer.preflight_summary(), indent=2, sort_keys=True))
        return
    trainer.train()


if __name__ == "__main__":
    main()

