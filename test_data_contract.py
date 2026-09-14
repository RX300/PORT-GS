"""CPU checks for the explicit PORT-GS metadata split contract."""

import json
import unittest
from pathlib import Path
from unittest import mock

import torch

import data as data_module
from data import SceneDataset


DATA_ROOT = Path("/workspace/datasets/SSD-GS/data")


class TestDataContract(unittest.TestCase):
    """Check JSON selection without touching validation or test images."""

    def _load_dataset_metadata(
        self,
        scene: Path,
        split: str,
        metadata_name: str,
        unit_light_intensity: float | None = None,
    ) -> tuple[SceneDataset, dict]:
        expected_path = scene / metadata_name
        with expected_path.open("r", encoding="utf-8") as handle:
            expected = json.load(handle)

        kwargs = {"scene_path": scene, "split": split}
        if unit_light_intensity is not None:
            kwargs["unit_light_intensity"] = unit_light_intensity

        # Dataset construction is metadata-only.  A decode here would violate
        # the split-isolation contract and make the test read an image.
        with mock.patch.object(
            data_module,
            "_decode_image",
            side_effect=AssertionError("metadata construction decoded an image"),
        ):
            dataset = SceneDataset(**kwargs)

        self.assertEqual(dataset.metadata_path.resolve(), expected_path.resolve())
        self.assertEqual(dataset.metadata, expected)
        self.assertEqual(dataset.frames, expected["frames"])

        # Compare every referenced path and its file-extension metadata, not a
        # count or a selected index.
        def path_records(frames: list[dict]) -> list[tuple]:
            records = []
            for frame in frames:
                if "file_paths" in frame:
                    records.append(("file_paths", tuple(frame["file_paths"])))
                else:
                    records.append(
                        ("file_path", frame["file_path"], frame["file_ext"])
                    )
            return records

        self.assertEqual(path_records(dataset.frames), path_records(expected["frames"]))
        return dataset, expected

    def test_explicit_split_files_and_complete_records(self) -> None:
        cases = (
            (
                DATA_ROOT / "Real_NRHints" / "Pixiu",
                {
                    "train": "transforms_train.json",
                    "val": "transforms_valid.json",
                    "test": "transforms_test.json",
                },
                None,
            ),
            (
                DATA_ROOT / "Synthetic_GS3" / "Translucent",
                {
                    "train": "transforms_train.json",
                    "test": "transforms_test.json",
                },
                None,
            ),
            (
                DATA_ROOT / "Synthetic_SSS-GS" / "bunny_small",
                {
                    "train": "transforms_train.json",
                    "val": "transforms_val.json",
                    "test": "transforms_test.json",
                },
                1.0,
            ),
        )

        datasets = {}
        for scene, split_files, unit_light_intensity in cases:
            self.assertTrue(scene.is_dir(), scene)
            for split, metadata_name in split_files.items():
                dataset, _ = self._load_dataset_metadata(
                    scene, split, metadata_name, unit_light_intensity
                )
                datasets[(scene, split)] = dataset

        # GS3 has no validation JSON; requesting it must fail explicitly rather
        # than silently aliasing train or test metadata.
        gs3 = DATA_ROOT / "Synthetic_GS3" / "Translucent"
        self.assertFalse((gs3 / "transforms_val.json").exists())
        with mock.patch.object(
            data_module,
            "_decode_image",
            side_effect=AssertionError("metadata construction decoded an image"),
        ):
            with self.assertRaises(FileNotFoundError):
                SceneDataset(gs3, split="val")

        self._check_one_sss_train_sample(
            datasets[(DATA_ROOT / "Synthetic_SSS-GS" / "bunny_small", "train")]
        )

    def _check_one_sss_train_sample(self, dataset: SceneDataset) -> None:
        """Decode exactly one correct train image and check its calibration."""

        scene = dataset.scene_path
        with (scene / "train" / "anno.json").open("r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        frame = dataset.frames[0]
        annotation = annotations[0]
        self.assertEqual(Path(frame["file_paths"][0]).stem, Path(annotation["filename"]).stem)

        with mock.patch.object(
            data_module, "_decode_image", wraps=data_module._decode_image
        ) as decode:
            sample = dataset[0]
        self.assertEqual(decode.call_count, 1)

        expected_c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        torch.testing.assert_close(sample["c2w"], expected_c2w)
        flip = torch.diag(torch.tensor((1.0, -1.0, -1.0, 1.0)))
        expected_viewmat = torch.linalg.inv(expected_c2w @ flip)
        torch.testing.assert_close(sample["viewmat"], expected_viewmat)
        for row in range(3):
            for column in range(4):
                self.assertAlmostEqual(
                    float(sample["viewmat"][row, column]),
                    float(annotation["RT"][row][column]),
                    places=5,
                )

        # The official SSS-GS reader flips Y/Z once while making CameraInfo and
        # CameraDataset flips them again.  The endpoint and anno.json are raw.
        for actual, expected in zip(sample["light_pos"].tolist(), annotation["light_pos"]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(sample["light_intensity"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(sample["frame_index"], 0)
        self.assertEqual(sample["name"], Path(frame["file_paths"][0]).stem)


if __name__ == "__main__":
    unittest.main()
