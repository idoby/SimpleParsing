from argparse import Namespace
from typing import Any, Callable, Dict, Iterable, List, Sequence, Type, TypeVar, Union
from simple_parsing import ArgumentParser
from simple_parsing.utils import Dataclass
from dataclasses import dataclass
import copy
import sys


def add_subparsers(
    parser: ArgumentParser,
    title: str,
    name_to_type: Dict[str, Type[Dataclass]],
    dest: str = None,
    metavar: str = None,
    required: bool = True,
    name_to_help: Dict[str, str] = None,
) -> Dict[str, ArgumentParser]:
    # Dict to be returned.
    subparsers: Dict[str, ArgumentParser] = {}

    # Keyword for the `add_subparsers` function, which differ a bit based on the version of python.
    kwargs = dict(title=title, dest=dest or "", metavar=metavar)
    if sys.version_info >= (3, 7):
        kwargs["required"] = required
    subparser = parser.add_subparsers(**kwargs)

    for name, type in name_to_type.items():
        dest = dest or title
        help = name_to_help[name] if name_to_help else type.__doc__
        description = f"FIXME: Description for {type}: {type.__doc__}"
        type_parser: ArgumentParser = subparser.add_parser(name, help=help, description=description)

        # NOTE: If we wanted to give access to the 'context' when adding arguments for this
        # subparser, we could do this here like so:
        # type_parser._parent = parser
        #
        # def get_parents(parser: ArgumentParser) -> List[ArgumentParser]:
        #     parents = []
        #     while hasattr(parser, "_parent"):
        #         parents.append(parser._parent)
        #         parser = parser._parent
        #     parents.reverse()
        #     return parents
        #         type_parser.add_arguments(type, dest=dest)
        #         subparsers[name] = type_parser
        #     return subparsers
        #
        # parsers = get_parents(type_parser) + [type_parser]
        # context = sum([[w.dataclass.__qualname__ for w in p._wrappers] for p in parsers], [])
        # def add_arguments_fn(parser: ArgumentParser, type: Type[Dataclass], dest: str, context: Any) -> None:
        #     parser.add_arguments(type, dest=dest)


@dataclass
class Letter:
    """Some kind of letter."""

    uppercase: bool = False


@dataclass
class A(Letter):
    """The great letter 'a', the first in the alphabet."""

    # An argument specic to the letter 'a'.
    foo_a: int = 1


@dataclass
class B(Letter):
    """The meh letter 'b', the second one the alphabet."""

    # An argument specic to the letter 'b'.
    foo_b: int = 1


@dataclass
class C(Letter):
    """The ugly letter 'c', the third one the alphabet."""

    # An argument specic to the letter 'c'.
    foo_c: int = 1


@dataclass
class Person:
    """ Some person. """

    # name of the person.
    name: str = ""


@dataclass
class Bob(Person):
    """ Bob the builder. A great person. """

    # Name for bob.
    name: str = "Bob"
    # Some argument specific to Bob.
    foo_bob: int = 1


@dataclass
class Alice(Person):
    """ Alice in wonderland. """

    # Alice's name.
    name: str = "Alice"
    # Some argument specific to Alice.
    foo_alice: int = 1


@dataclass
class Claire(Person):
    """ My aunt Claire. """

    # Claire's name.
    name: str = "Claire"
    # Some argument specific to Claire.
    foo_claire: int = 1


def main_simple_example():
    parser = ArgumentParser("demo")

    letters = {"a": A, "b": B, "c": C}
    persons = {"bob": Bob, "claire": Claire, "alice": Alice}

    child_parsers = add_subparsers(
        parser,
        title="letter_or_person",
        name_to_type={**letters, **persons},
        # Storing the result of the "first" subparsers in a temp attribute and move it after.
        dest="temp",
    )

    for name, child_parser in child_parsers.items():
        # Add the subparsers for the other group.
        if name in letters:
            add_subparsers(child_parser, title="person", dest="person", name_to_type=persons)
        elif name in persons:
            add_subparsers(child_parser, title="letter", dest="letter", name_to_type=letters)

    args = parser.parse_args()

    # post-processing:
    if isinstance(args.temp, tuple(letters.values())):
        args.letter = args.temp
        delattr(args, "temp")
    elif isinstance(args.temp, tuple(persons.values())):
        args.person = args.temp
        delattr(args, "temp")
    else:
        assert False, args

    print(args)


@dataclass
class Flower:
    """ Some flower. """

    # How intense the smell is, from 0 to 10.
    smell_intensity: int = 0


@dataclass
class Rose(Flower):
    """ A really pretty flower that everyone likes. """

    # Roses have a pleasant, delicate smell to them.
    smell_intensity: int = 2
    # If these roses are meant as a romantic gift or not.
    romantic: bool = True


@dataclass
class Petunia(Flower):
    """ A flower I don't actually know anything about. """

    # Fake intensity for petunias.
    smell_intensity: int = 3
    # Some attribute specific to them
    count: int = 123


def main_complicated():
    """ More complicated demo that works with any number of groups of subparsers."""
    parser = ArgumentParser("demo_complicated")

    postprocessing_fn = independant_subparsers(
        parser,
        letter={"a": A, "b": B, "c": C},
        person={"bob": Bob, "claire": Claire, "alice": Alice},
        flower={"rose": Rose, "petunia": Petunia},
    )

    args = parser.parse_args()

    # Postprocessing: need to move the values from their placeholder dests to their intended dests.
    # args = postprocessing_fn(args)

    print(args)


def independant_subparsers(
    parser: ArgumentParser,
    _level: int = 0,
    **group_name_to_subparser_dict: Dict[str, Type[Dataclass]],
) -> Callable[[Namespace], Namespace]:
    """Add multiple independant subparsers to the given parser.

    This works by adding subparsers in a hierarchical fashion.

    NOTE: This can only be called once for any given parser/subparser.
    """
    if not group_name_to_subparser_dict:
        raise ValueError("Need non-empty mapping from group name to subparser types.")

    n_groups = len(group_name_to_subparser_dict)

    title = " or ".join(group_name_to_subparser_dict.keys())
    metavar = "|".join(
        f"<{group_name}>"
        # f"<{group_name}>:" + "{" + ",".join(names_to_types.keys()) + "}"
        for group_name, names_to_types in group_name_to_subparser_dict.items()
    )

    # Get a dict with the types of each subparser for the "first" command.
    name_to_type: Dict[str, Type[Dataclass]] = {}
    for group_name, names_to_types in group_name_to_subparser_dict.items():
        for name, type in names_to_types.items():
            if name in name_to_type:
                raise RuntimeError(f"Conflicting name: {name} is already")()
        name_to_type.update(names_to_types)

    child_parsers = add_subparsers(
        parser,
        title=title,
        name_to_type=name_to_type,
        metavar=metavar,
        dest=f"command_{_level}",
    )

    # There are other subparser groups for which we haven't yet added subparsers for: recurse!
    for name, child_parser in child_parsers.items():
        # For each child subparser we created above, we need to add arguments for all the other
        # if there are any subparsers groups left, we add them.
        for group_name, name_to_types in group_name_to_subparser_dict.items():
            # Check if the child parser belongs to that group.
            if name in name_to_types.keys():
                child_parser.set_defaults(**{f"command_{_level}_dest": group_name})

                rest = group_name_to_subparser_dict.copy()
                rest.pop(group_name)
                if rest:
                    # NOTE: We don't use the result of this recursive call, because the
                    # postprocessing_fn below handles all those.
                    _ = independant_subparsers(child_parser, **rest, _level=_level + 1)

    def postprocessing_fn(args: Namespace, inplace: bool = False) -> Namespace:
        """Function to be applied to the args after they are parsed to fix the arguments being at
        the wrong destinations.
        """
        if not inplace:
            args = copy.deepcopy(args)

        for level in range(n_groups):
            # Extract the destination and delete that temporary attribute.
            command_dest = getattr(args, f"command_{level}_dest")
            delattr(args, f"command_{level}_dest")

            # Extract the value and delete that temporary attribute.
            command_value = getattr(args, f"command_{level}")
            delattr(args, f"command_{level}")

            # Set the value at the intended attribute.
            if hasattr(args, command_dest):
                raise RuntimeError(
                    f"Expected args to not yet have a value at attribute {command_dest}"
                )

            setattr(args, command_dest, command_value)

        return args

    if _level == 0:
        _original = parser.parse_args
        parser.parse_args = lambda *args, **kwargs: postprocessing_fn(_original(*args, **kwargs))
    return postprocessing_fn


if __name__ == "__main__":
    # main_simple_example()
    main_complicated()
