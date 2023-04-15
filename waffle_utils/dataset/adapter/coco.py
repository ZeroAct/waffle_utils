# elif data_type == DataType.COCO_DETECTION:
#             """COCO DETECTION FORMAT
#             - directory format
#                 coco_dataset/
#                     images/
#                         1.png
#                         ...
#                     train.json
#                     val.json
#                     test.json
#             """

#             def _export(images: list[Image], set_name: str, export_dir: Path):
#                 image_dir = export_dir / "images"
#                 label_path = export_dir / f"{set_name}.json"

#                 io.make_directory(image_dir)

#                 coco = {"categories": [], "images": [], "annotations": []}

#                 for category in self.get_categories():
#                     d = category.to_dict()
#                     category_id = d.pop("category_id")
#                     coco["categories"].append({"id": category_id, **d})

#                 for image in images:
#                     image_path = self.raw_image_dir / image.file_name
#                     image_dst_path = image_dir / image.file_name
#                     io.copy_file(
#                         image_path, image_dst_path, create_directory=True
#                     )

#                     d = image.to_dict()
#                     image_id = d.pop("image_id")
#                     coco["images"].append({"id": image_id, **d})

#                     annotations = self.get_annotations(image_id)
#                     for annotation in annotations:
#                         d = annotation.to_dict()
#                         annotation_id = d.pop("annotation_id")
#                         coco["annotations"].append({"id": annotation_id, **d})

#                 io.save_json(coco, label_path, create_directory=True)

#             if train_image_ids:
#                 _export(self.get_images(train_image_ids), "train", export_dir)
#             if val_image_ids:
#                 _export(self.get_images(val_image_ids), "val", export_dir)
#             if test_image_ids:
#                 _export(self.get_images(test_image_ids), "test", export_dir)
#             if unlabeled_image_ids:
#                 _export(
#                     self.get_images(unlabeled_image_ids, labeled=False),
#                     "unlabeled",
#                     export_dir,
#                 )

#             return str(export_dir)
