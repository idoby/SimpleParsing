from argparse import Namespace
from typing import Dict, Type
from simple_parsing import ArgumentParser
from simple_parsing.utils import Dataclass
from dataclasses import dataclass
import copy
import sys


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


def add_subparsers(
    parser: ArgumentParser,
    title: str,
    name_to_type: Dict[str, Type[Dataclass]],
    dest: str = None,
    metavar: str = None,
    required: bool = True,
) -> Dict[str, ArgumentParser]:
    """ Add subparsers to the given parser, one per dataclass type. """
    # Dict to be returned.
    subparsers: Dict[str, ArgumentParser] = {}

    # Keyword for the `add_subparsers` function, which differ a bit based on the version of python.
    kwargs = dict(title=title, dest=dest or "", metavar=metavar)
    if sys.version_info >= (3, 7):
        kwargs["required"] = required
    subparser = parser.add_subparsers(**kwargs)

    for name, type in name_to_type.items():
        dest = dest or title
        # NOTE: Could perhaps do something smarter than this.
        help = next(line for line in type.__doc__.splitlines() if not line.isspace())
        description = type.__doc__
        type_parser: ArgumentParser = subparser.add_parser(
            name,
            help=help,
            description=description,
        )
        type_parser.add_arguments(type, dest=dest)
        subparsers[name] = type_parser
    return subparsers


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

    # post-processing: Rearrange the arguments.
    if isinstance(args.temp, Letter):
        args.letter = args.temp
        delattr(args, "temp")
    elif isinstance(args.temp, Person):
        args.person = args.temp
        delattr(args, "temp")

    print(args)


# ---------- Now: onto something a bit more challenging: What if we add a third group? ------------


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

    add_independent_subparsers(
        parser,
        letter={"a": A, "b": B, "c": C},
        person={"bob": Bob, "claire": Claire, "alice": Alice},
        flower={"rose": Rose, "petunia": Petunia},
    )

    args = parser.parse_args()
    print(args)


def add_independent_subparsers(
    parser: ArgumentParser,
    _level: int = 0,
    **group_name_to_subparser_dict: Dict[str, Type[Dataclass]],
) -> None:
    """Add multiple independent subparsers to the given parser.

    This works by adding subparsers in a hierarchical fashion.

    NOTE: This can only be called once for any given parser/subparser.
    """
    if not group_name_to_subparser_dict:
        raise ValueError("Need non-empty mapping from group name to subparser types.")

    # TODO: Leaving the 'optional multiple subparsers' feature for future work.
    required: bool = True

    n_groups = len(group_name_to_subparser_dict)

    title = " or ".join(group_name_to_subparser_dict.keys())
    metavar = "|".join(
        f"<{group_name}>"
        for group_name in group_name_to_subparser_dict.keys()
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
        required=required,
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
                    _ = add_independent_subparsers(child_parser, **rest, _level=_level + 1)

    def postprocessing_fn(args: Namespace, inplace: bool = False) -> Namespace:
        """Function to be applied to the args after they are parsed to fix the arguments being at
        the wrong destinations.
        """
        if not inplace:
            args = copy.deepcopy(args)

        for level in range(n_groups):
            
            # Extract the destination and delete that temporary attribute.
            if not required:
                # TODO: The attribute might not be there, in which case we can skip this?
                if not hasattr(args, f"command_{level}_dest"):
                    continue

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
        # NOTE: A bit hacky, but keeps the API the same and makes it very convenient!
        # If at the 'root' level, then we modify the parser in-place to avoid having to call
        # the postprocessing_fn manually:
        _parse_known_args = parser.parse_known_args

        def _wrapped_parse_known_args(*args, **kwargs):
            args, extra_args = _parse_known_args(*args, **kwargs)
            return postprocessing_fn(args), extra_args

        parser.parse_known_args = _wrapped_parse_known_args


if __name__ == "__main__":
    # main_simple_example()
    main_complicated()
