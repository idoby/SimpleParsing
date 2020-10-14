import copy
import inspect
import json
import warnings
from collections import OrderedDict
from dataclasses import MISSING, Field, asdict, dataclass, fields, is_dataclass
from functools import singledispatch
from pathlib import Path
from typing import *
from typing import IO, TypeVar

import typing_inspect as tpi

from ...logging_utils import get_logger
from ...utils import get_type_arguments, is_dict, is_list, is_union
from .decoding import (_decoding_fns, decode_field, get_decoding_fn,
                       register_decoding_fn)
from .encoding import SimpleJsonEncoder, encode

logger = get_logger(__file__)

Dataclass = TypeVar("Dataclass")
D = TypeVar("D", bound="FrozenSerializable")

try:
    import yaml
    def ordered_dict_constructor(loader: yaml.Loader, node: yaml.Node):
        value = loader.construct_sequence(node)
        return OrderedDict(*value)

    def ordered_dict_representer(dumper: yaml.Dumper, instance: OrderedDict) -> yaml.Node:
        node = dumper.represent_sequence("OrderedDict", instance.items())
        return node

    yaml.add_representer(OrderedDict, ordered_dict_representer)
    yaml.add_constructor("OrderedDict", ordered_dict_constructor)
    yaml.add_constructor("tag:yaml.org,2002:python/object/apply:collections.OrderedDict", ordered_dict_constructor)

except ImportError:
    pass

from .serializable import from_dict, get_init_fields, get_first_non_None_type

@dataclass(frozen=True)
class FrozenSerializable:
    """Makes a dataclass serializable to and from dictionaries.

    Supports JSON and YAML files for now.

    >>> from dataclasses import dataclass
    >>> from simple_parsing.helpers import FrozenSerializable
    >>> @dataclass(frozen=True)
    ... class Config(FrozenSerializable):
    ...   a: int = 123
    ...   b: str = "456"
    ... 
    >>> config = Config()
    >>> config
    Config(a=123, b='456')
    >>> config.to_dict()
    {"a": 123, "b": 456}
    >>> config_ = Config.from_dict({"a": 123, "b": 456})
    Config(a=123, b='456')
    >>> assert config == config_
    """
    subclasses: ClassVar[List[Type[D]]] = []
    decode_into_subclasses: ClassVar[bool] = False

    def __init_subclass__(cls, decode_into_subclasses: bool=None, add_variants: bool=True):
        logger.debug(f"Registering a new FrozenSerializable subclass: {cls}")
        if decode_into_subclasses is None:
            # if decode_into_subclasses is None, we will use the value of the
            # parent class, if it is also a subclass of FrozenSerializable.
            # Skip the class itself as well as object.
            parents = cls.mro()[1:-1]
            logger.debug(f"parents: {parents}")

            for parent in parents:
                if parent in FrozenSerializable.subclasses and parent is not FrozenSerializable:
                    decode_into_subclasses = parent.decode_into_subclasses
                    logger.debug(f"Parent class {parent} has decode_into_subclasses = {decode_into_subclasses}")
                    break
        super().__init_subclass__()

        cls.decode_into_subclasses = decode_into_subclasses or False
        if cls not in FrozenSerializable.subclasses:
            FrozenSerializable.subclasses.append(cls)

        encode.register(cls, cls.to_dict)
        register_decoding_fn(cls, cls.from_dict)

    def to_dict(self, dict_factory:Type[Dict]=dict, recurse: bool=True) -> Dict:
        """ Serializes this dataclass to a dict.
        
        NOTE: This 'extends' the `asdict()` function from
        the `dataclasses` package, allowing us to not include some fields in the
        dict, or to perform some kind of custom encoding (for instance,
        detaching `Tensor` objects before serializing the dataclass to a dict).
        """
        d: Dict[str, Any] = dict_factory()
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            T = f.type

            # Do not include in dict if some corresponding flag was set in metadata.
            include_in_dict = f.metadata.get("to_dict", True)
            if not include_in_dict:
                continue

            custom_encoding_fn = f.metadata.get("encoding_fn")
            if custom_encoding_fn:
                # Use a custom encoding function if there is one.
                d[name] = custom_encoding_fn(value)
                continue

            encoding_fn = encode
            if isinstance(value, FrozenSerializable) and recurse:
                try:
                    encoded = value.to_dict(dict_factory=dict_factory, recurse=recurse)
                except TypeError as e:
                    encoded = value.to_dict()
                logger.debug(f"Encoded FrozenSerializable field {name}: {encoded}")
            else:
                try:
                    encoded = encoding_fn(value)
                except Exception as e:
                    logger.error(f"Unable to encode value {value} of type {type(value)}! Leaving it as-is. (exception: {e})")
                    encoded = value
            d[name] = encoded
        return d

    @classmethod
    def from_dict(cls: Type[D], obj: Dict, drop_extra_fields: bool=None) -> D:
        """ Parses an instance of `cls` from the given dict.
        
        NOTE: If the `decode_into_subclasses` class attribute is set to True (or
        if `decode_into_subclasses=True` was passed in the class definition),
        then if there are keys in the dict that aren't fields of the dataclass,
        this will decode the dict into an instance the first subclass of `cls`
        which has all required field names present in the dictionary.
        
        Passing `drop_extra_fields=None` (default) will use the class attribute
        described above.
        Passing `drop_extra_fields=True` will decode the dict into an instance
        of `cls` and drop the extra keys in the dict.
        Passing `drop_extra_fields=False` forces the above-mentioned behaviour.
        """
        if drop_extra_fields is None:
            drop_extra_fields = not cls.decode_into_subclasses
        return from_dict(cls, obj, drop_extra_fields=drop_extra_fields, Serializable=FrozenSerializable)

    def dump(self, fp: IO[str], dump_fn=json.dump, **kwargs) -> None:
        # Convert `self` into a dict.
        d = self.to_dict()
        # Serialize that dict to the file, using dump_fn.
        dump_fn(d, fp, **kwargs)
    
    def dump_json(self, fp: IO[str], dump_fn=json.dump, **kwargs) -> str:
        return self.dump(fp, dump_fn=dump_fn, **kwargs)

    def dump_yaml(self, fp: IO[str], dump_fn=None, **kwargs) -> str:
        import yaml
        if dump_fn is None:
            dump_fn = yaml.dump
        return self.dump(fp, dump_fn=dump_fn, **kwargs)

    def dumps(self, dump_fn=json.dumps, **kwargs) -> str:
        d = self.to_dict()
        return dump_fn(d, **kwargs)

    def dumps_json(self, dump_fn=json.dumps, **kwargs) -> str:
        kwargs.setdefault("cls", SimpleJsonEncoder)
        return self.dumps(dump_fn=dump_fn, **kwargs)

    def dumps_yaml(self, dump_fn=None, **kwargs) -> str:
        import yaml
        if dump_fn is None:
            dump_fn = yaml.dump
        return self.dumps(dump_fn=dump_fn, **kwargs)

    @classmethod
    def load(cls: Type[D], path: Union[Path, str, IO[str]], drop_extra_fields: bool=None, load_fn=None, **kwargs) -> D:
        """Loads an instance of `cls` from the given file.
        
        Args:
            cls (Type[D]): A dataclass type to load.
            path (Union[Path, str, IO[str]]): Path or Path string or open file.
            drop_extra_fields (bool, optional): Wether to drop extra fields or 
                to decode the dictionary into the first subclass with matching
                fields. Defaults to None, in which case we use the value of
                `cls.decode_into_subclasses`. 
                For more info, see `cls.from_dict`.
            load_fn ([type], optional): Which loading function to use. Defaults
                to None, in which case we try to use the appropriate loading
                function depending on `path.suffix`:
                {
                    ".yml": yaml.full_load,
                    ".yaml": yaml.full_load,
                    ".json": json.load,
                    ".pth": torch.load,
                    ".pkl": pickle.load,
                }

        Raises:
            RuntimeError: If the extension of `path` is unsupported.

        Returns:
            D: An instance of `cls`.
        """
        if isinstance(path, str):
            path = Path(path)

        if load_fn is None and isinstance(path, Path):
            if path.name.endswith((".yml", ".yaml")):
                return cls.load_yaml(path, drop_extra_fields=drop_extra_fields, **kwargs)
            elif path.name.endswith(".json"):
                return cls.load_json(path, drop_extra_fields=drop_extra_fields, **kwargs)
            elif path.name.endswith(".pth"):
                import torch
                load_fn = torch.loads
            elif path.name.endswith(".npy"):
                import numpy as np
                load_fn = np.load
            elif path.name.endswith(".pkl"):
                import pickle
                load_fn = pickle.load
            warnings.warn(RuntimeWarning(
                f"Not sure how to deserialize contents of {path} to a dict, as no "
                f" load_fn was passed explicitly. Will try to use {load_fn} as the "
                f"load function, based on the path name."    
            )) 

        if load_fn is None:
            raise RuntimeError(
                f"Unable to determine what function to use in order to load "
                f"path {path} into a dictionary, since no load_fn was passed, "
                f"and the path doesn't have a familiar extension.."
            )

        if isinstance(path, Path):
            path = path.open()
        return cls._load(path, load_fn=load_fn, drop_extra_fields=drop_extra_fields, **kwargs)

    @classmethod
    def _load(cls: Type[D], fp: IO[str], drop_extra_fields: bool=None, load_fn=json.load, **kwargs) -> D:
        # Load a dict from the file.
        d = load_fn(fp, **kwargs)
        # Convert the dict into an instance of the class.
        return cls.from_dict(d, drop_extra_fields=drop_extra_fields)

    @classmethod
    def load_json(cls: Type[D], path: Union[str, Path], drop_extra_fields: bool=None, load_fn=json.load, **kwargs) -> D:
        """Loads an instance from the corresponding json-formatted file.

        Args:
            cls (Type[D]): A dataclass type to load.
            path (Union[str, Path]): Path to a json-formatted file.
            load_fn ([type], optional): Loading function to use. Defaults to json.load.

        Returns:
            D: an instance of the dataclass.
        """
        return cls.load(path, drop_extra_fields=drop_extra_fields, load_fn=load_fn, **kwargs)

    @classmethod
    def load_yaml(cls: Type[D], path: Union[str, Path], drop_extra_fields: bool=None, load_fn=None, **kwargs) -> D:
        """Loads an instance from the corresponding yaml-formatted file.

        Args:
            cls (Type[D]): A dataclass type to load.
            path (Union[str, Path]): Path to a yaml-formatted file.
            load_fn ([type], optional): Loading function to use. Defaults to
                None, in which case `yaml.full_load` is used.

        Returns:
            D: an instance of the dataclass.
        """
        import yaml
        if load_fn is None:
            load_fn = yaml.full_load
        return cls.load(path, load_fn=load_fn, drop_extra_fields=drop_extra_fields, **kwargs)

    def save(self, path: Union[str, Path], dump_fn=None, **kwargs) -> None:
        if not isinstance(path, Path):
            path = Path(path)

        if dump_fn is None and isinstance(path, Path):
            if path.name.endswith((".yml", ".yaml")):
                return self.save_yaml(path, **kwargs)
            elif path.name.endswith(".json"):
                return self.save_json(path, **kwargs)
            elif path.name.endswith(".pth"):
                import torch
                dump_fn = torch.save
            elif path.name.endswith(".npy"):
                import numpy as np
                dump_fn = np.save
            elif path.name.endswith(".pkl"):
                import pickle
                dump_fn = pickle.dump
            warnings.warn(RuntimeWarning(
                f"Not 100% sure how to deserialize contents of {path} to a "
                f"file as no dump_fn was passed explicitly. Will try to use "
                f"{dump_fn} as the serialization function, based on the path "
                f"suffix. ({path.suffix})" 
            ))

        if dump_fn is None:
            raise RuntimeError(
                f"Unable to determine what function to use in order to dump "
                f"path {path} into a dictionary, since no dump_fn was passed, "
                f"and the path doesn't have an unfamiliar extension: "
                f"({path.suffix})"
            )
        self._save(path, dump_fn=dump_fn, **kwargs)

    def _save(self, path: Union[str, Path], dump_fn=json.dump, **kwargs) -> None:    
        d = self.to_dict()
        logger.debug(f"saving to path {path}")
        with open(path, "w") as fp:
            dump_fn(d, fp, **kwargs)

    def save_yaml(self, path: Union[str, Path], dump_fn=None, **kwargs) -> None:
        import yaml
        if dump_fn is None:
            dump_fn = yaml.dump
        self.save(path, dump_fn=dump_fn, **kwargs)

    def save_json(self, path: Union[str, Path], dump_fn=json.dump, **kwargs) -> None:
        self.save(path, dump_fn=dump_fn, **kwargs)


    @classmethod
    def loads(cls: Type[D], s: str, drop_extra_fields: bool=None, load_fn=json.loads, **kwargs) -> D:
        d = load_fn(s, **kwargs)
        return cls.from_dict(d, drop_extra_fields=drop_extra_fields)

    @classmethod
    def loads_json(cls: Type[D], s: str, drop_extra_fields: bool=None, load_fn=json.loads, **kwargs) -> D:
        return cls.loads(s, drop_extra_fields=drop_extra_fields, load_fn=load_fn, **kwargs)

    @classmethod
    def loads_yaml(cls: Type[D], s: str, drop_extra_fields: bool=None, load_fn=None, **kwargs) -> D:
        import yaml
        if load_fn is None:
            load_fn = yaml.full_load
        return cls.loads(s, drop_extra_fields=drop_extra_fields, load_fn=load_fn, **kwargs)


@dataclass(frozen=True)
class SimpleFrozenSerializable(FrozenSerializable, decode_into_subclasses=True):
    pass


def get_dataclass_type_from_forward_ref(forward_ref: Type, FrozenSerializable=FrozenSerializable) -> Optional[Type]:
    arg = tpi.get_forward_arg(forward_ref)
    potential_classes: List[Type] = []
    
    for serializable_class in FrozenSerializable.subclasses:
        if serializable_class.__name__ == arg:
            potential_classes.append(serializable_class)

    if not potential_classes:
        logger.warning(
            f"Unable to find a corresponding type for forward ref "
            f"{forward_ref} inside the registered {FrozenSerializable} subclasses. "
            f"(Consider adding {FrozenSerializable} as a base class to <{arg}>? )."
        )
        return None
    elif len(potential_classes) > 1:
        logger.warning(
            f"More than one potential {FrozenSerializable} subclass was found for "
            f"forward ref '{forward_ref}'. The appropriate dataclass will be "
            f"selected based on the matching fields. \n"
            f"Potential classes: {potential_classes}"
        )
        return FrozenSerializable
    else:
        assert len(potential_classes) == 1
        return potential_classes[0]


def get_actual_type(field_type: Type) -> Type:
    if is_union(field_type):
        logger.debug(f"field has union type: {field_type}")
        t = get_first_non_None_type(field_type)
        logger.debug(f"First non-none type: {t}")
        if t is not None:
            field_type = t

    if tpi.is_forward_ref(field_type):
        logger.debug(f"field_type {field_type} is a forward ref.")
        dc = get_dataclass_type_from_forward_ref(field_type)
        logger.debug(f"Found the corresponding type: {dc}")
        if dc is not None:
            field_type = dc
    return field_type
