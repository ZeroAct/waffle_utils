import warnings

from waffle_utils.dataset import (
    Annotation,
    Category,
    Dataset,
    DataType,
    Image,
    TaskType,
)
from waffle_utils.file import io

# if data_type == DataType.YOLO:
#     f"""YOLO DETECTION FORMAT
#     - directory format
#         yolo_dataset/
#             train/
#                 images/
#                     1.png
#                 labels/
#                     1.txt
#                     ```
#                     class x_center y_center width height
#                     ```
#             val/
#                 images/
#                     2.png
#                 labels/
#                     2.txt
#             test/
#                 images/
#                     3.png
#                 labels/
#                     3.txt

#     - dataset.yaml
#         path: [dataset_dir]/exports/{data_type.name}
#         train: train
#         val: val
#         names:
#             0: person
#             1: bicycle
#             ...
#     """

#     def _export(images: list[Image], export_dir: Path):
#         image_dir = export_dir / "images"
#         label_dir = export_dir / "labels"

#         io.make_directory(image_dir)
#         io.make_directory(label_dir)

#         for image in images:
#             image_path = self.raw_image_dir / image.file_name
#             image_dst_path = image_dir / image.file_name
#             label_dst_path = (label_dir / image.file_name).with_suffix(
#                 ".txt"
#             )
#             io.copy_file(
#                 image_path, image_dst_path, create_directory=True
#             )

#             W = image.width
#             H = image.height

#             annotations: list[Annotation] = self.get_annotations(
#                 image.image_id
#             )
#             label_txts = []
#             for annotation in annotations:
#                 x1, y1, w, h = annotation.bbox
#                 x1, w = x1 / W, w / W
#                 y1, h = y1 / H, h / H
#                 cx, cy = x1 + w / 2, y1 + h / 2

#                 category_id = annotation.category_id - 1

#                 label_txts.append(f"{category_id} {cx} {cy} {w} {h}")

#             io.make_directory(label_dst_path.parent)
#             with open(label_dst_path, "w") as f:
#                 f.write("\n".join(label_txts))

#     if train_image_ids:
#         io.make_directory(export_dir / "train")
#         _export(self.get_images(train_image_ids), export_dir / "train")
#     if val_image_ids:
#         io.make_directory(export_dir / "val")
#         _export(self.get_images(val_image_ids), export_dir / "val")
#     if test_image_ids:
#         io.make_directory(export_dir / "test")
#         _export(self.get_images(test_image_ids), export_dir / "test")

#     io.save_yaml(
#         {
#             "path": str(export_dir.absolute()),
#             "train": "train",
#             "val": "val",
#             "test": "test",
#             "names": {
#                 category.category_id - 1: category.name
#                 for category in self.get_categories()
#             },
#         },
#         export_dir / "data.yaml",
#     )

#     return str(export_dir)

# elif data_type == DataType.YOLO_CLASSIFICATION:
#     f"""YOLO CLASSIFICATION FORMAT (compatiable with torchvision.datasets.ImageFolder)
#     - directory format
#         yolo_dataset/
#             train/
#                 person/
#                     1.png
#                 bicycle/
#                     2.png
#             val/
#                 person/
#                     3.png
#                 bicycle/
#                     4.png
#             test/
#                 person/
#                     5.png
#                 bicycle/
#                     6.png
#     - dataset.yaml
#         path: [dataset_dir]/exports/{data_type.name}
#         train: train
#         val: val
#         test: test
#         names:
#             0: person
#             1: bicycle
#             ...
#     """

#     def _export(
#         images: list[Image],
#         categories: list[Category],
#         export_dir: Path,
#     ):
#         image_dir = export_dir
#         cat_dict: dict = {
#             cat.category_id: cat.name for cat in categories
#         }

#         for image in images:
#             image_path = self.raw_image_dir / image.file_name

#             annotations: list[Annotation] = self.get_annotations(
#                 image.image_id
#             )
#             if len(annotations) > 1:
#                 warnings.warn(
#                     f"Multi label does not support yet. Skipping {image_path}."
#                 )
#                 continue
#             category_id = annotations[0].category_id

#             image_dst_path = (
#                 image_dir / cat_dict[category_id] / image.file_name
#             )
#             io.copy_file(
#                 image_path, image_dst_path, create_directory=True
#             )

#     if train_image_ids:
#         io.make_directory(export_dir / "train")
#         _export(
#             self.get_images(train_image_ids),
#             self.get_categories(),
#             export_dir / "train",
#         )
#     if val_image_ids:
#         io.make_directory(export_dir / "val")
#         _export(
#             self.get_images(val_image_ids),
#             self.get_categories(),
#             export_dir / "val",
#         )
#     if test_image_ids:
#         io.make_directory(export_dir / "test")
#         _export(
#             self.get_images(test_image_ids),
#             self.get_categories(),
#             export_dir / "test",
#         )

#     io.save_yaml(
#         {
#             "path": str(export_dir.absolute()),
#             "train": "train",
#             "val": "val",
#             "test": "test",
#             "names": {
#                 category.category_id - 1: category.name
#                 for category in self.get_categories()
#             },
#         },
#         export_dir / "data.yaml",
#     )

#     return str(export_dir)

# elif data_type == DataType.YOLO_SEGMENTATION:
#     raise NotImplementedError


def export_yolo_classification(
    self: Dataset,
    export_dir: str,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    unlabeled_ids: list,
):
    """Export dataset to YOLO format for classification task

    Args:
        export_dir (str): Path to export directory
        train_ids (list): List of train ids
        val_ids (list): List of validation ids
        test_ids (list): List of test ids
        unlabeled_ids (list): List of unlabeled ids
    """
    io.make_directory(export_dir)

    categories: dict[int, Category] = self.categories

    for split, image_ids in zip(
        ["train", "val", "test", "unlabeled"],
        [train_ids, val_ids, test_ids, unlabeled_ids],
    ):
        if len(image_ids) == 0:
            continue

        split_dir = export_dir / split
        io.make_directory(split_dir)

        for image_id in image_ids:
            image = self.get_image(image_id)
            image_path = self.raw_image_dir / image.file_name

            annotations = self.get_annotations(image_id)
            if len(annotations) > 1:
                warnings.warn(
                    f"Multi label does not support yet. Skipping {image_path}."
                )
                continue
            category_id = annotations[0].category_id

            image_dst_path = (
                split_dir / categories[category_id].name / image.file_name
            )
            io.copy_file(image_path, image_dst_path, create_directory=True)


def export_yolo_detection(
    self: Dataset,
    export_dir: str,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    unlabeled_ids: list,
):
    """Export dataset to YOLO format for detection task

    Args:
        export_dir (str): Path to export directory
        train_ids (list): List of train ids
        val_ids (list): List of validation ids
        test_ids (list): List of test ids
        unlabeled_ids (list): List of unlabeled ids
    """
    io.make_directory(export_dir)

    image_to_annotations: dict[int, Annotation] = self.image_to_annotations

    for split, image_ids in zip(
        ["train", "val", "test", "unlabeled"],
        [train_ids, val_ids, test_ids, unlabeled_ids],
    ):
        if len(image_ids) == 0:
            continue

        image_dir = export_dir / split / "images"
        label_dir = export_dir / split / "labels"

        io.make_directory(image_dir)
        io.make_directory(label_dir)

        for image in self.get_images(image_ids):
            image_path = self.raw_image_dir / image.file_name
            image_dst_path = image_dir / image.file_name
            label_dst_path = (label_dir / image.file_name).with_suffix(".txt")
            io.copy_file(image_path, image_dst_path, create_directory=True)

            W = image.width
            H = image.height

            annotations: list[Annotation] = image_to_annotations[
                image.image_id
            ]
            label_txts = []
            for annotation in annotations:
                x1, y1, w, h = annotation.bbox
                x1, w = x1 / W, w / W
                y1, h = y1 / H, h / H
                cx, cy = x1 + w / 2, y1 + h / 2

                category_id = annotation.category_id - 1

                label_txts.append(f"{category_id} {cx} {cy} {w} {h}")

            io.make_directory(label_dst_path.parent)
            with open(label_dst_path, "w") as f:
                f.write("\n".join(label_txts))


def export_yolo(self: Dataset, export_dir: str):

    train_ids, val_ids, test_ids, unlabeled_ids = self.get_split_ids()

    if self.task == TaskType.CLASSIFICATION:
        export_dir = export_yolo_classification(
            self, export_dir, train_ids, val_ids, test_ids, unlabeled_ids
        )
    elif self.task == TaskType.OBJECT_DETECTION:
        export_dir = export_yolo_detection(
            self, export_dir, train_ids, val_ids, test_ids, unlabeled_ids
        )

    io.save_yaml(
        {
            "path": str(export_dir.absolute()),
            "train": "train",
            "val": "val",
            "test": "test",
            "names": {
                category_id - 1: category.name
                for category_id, category in self.categories.items()
            },
        },
        export_dir / "data.yaml",
    )

    return str(export_dir)
