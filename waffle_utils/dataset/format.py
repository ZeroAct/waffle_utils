import enum


class CustomEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        if isinstance(item, str):
            return item.upper() in cls._member_names_
        return super().__contains__(item)


class BaseEnum(enum.Enum, metaclass=CustomEnumMeta):
    """Base class for Enum

    Example:
        >>> class Color(BaseEnum):
        >>>     RED = 1
        >>>     GREEN = 2
        >>>     BLUE = 3
        >>> Color.RED == "red"
        True
        >>> Color.RED == "RED"
        True
        >>> "red" in DataType
        True
        >>> "RED" in DataType
        True
    """

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name.upper() == other.upper()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.name.upper())

    def __str__(self):
        return self.name.upper()

    def __repr__(self):
        return self.name.upper()


class DataType(BaseEnum):
    YOLO = enum.auto()
    COCO = enum.auto()


class TaskType(BaseEnum):
    CLASSIFICATION = enum.auto()
    OBJECT_DETECTION = enum.auto()
    SEGMENTATION = enum.auto()
    KEYPOINT_DETECTION = enum.auto()
    TEXT_RECOGNITION = enum.auto()
    REGRESSION = enum.auto()
