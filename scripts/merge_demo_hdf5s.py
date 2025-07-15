import glob
import pathlib
import time
from dataclasses import dataclass
from typing import List, Optional

import tyro

from demo_aug.utils.file_utils import count_total_demos, merge_demo_files


@dataclass
class Config:
    src_dir: Optional[str] = None
    src_paths: Optional[List[str]] = None
    save_path: Optional[str] = None

    def __post_init__(self):
        assert (
            self.src_paths is not None or self.src_dir is not None
        ), "src_dir or src_paths must be provided"
        if self.src_dir is not None:
            if self.src_paths is None:
                self.src_paths = []
            self.src_paths.extend(
                glob.glob(str(pathlib.Path(self.src_dir) / "**/*.hdf5"), recursive=True)
            )


def main(cfg: Config):
    start = time.time()
    if cfg.save_path is None:
        save_path = cfg.src_paths[0]
        total_demos = count_total_demos(cfg.src_paths)
        cfg.save_path = pathlib.Path(save_path).parent / (
            str(pathlib.Path(save_path).stem)
            + f"_{total_demos}demos"
            + str(pathlib.Path(save_path).suffix)
        )

    merge_demo_files(cfg.src_paths, pathlib.Path(cfg.save_path))
    end = time.time()
    print(f"took {end - start} seconds to merge {len(cfg.src_paths)} demos")


if __name__ == "__main__":
    tyro.cli(main)
