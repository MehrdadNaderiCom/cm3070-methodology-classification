"""Checks that catch the classes of defect found in the earlier prototype.

Each test here exists because the problem it looks for actually happened, either
in the prototype being rebuilt or during this rebuild.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cdfm.paths import REPO_ROOT

SOURCE_DIRECTORIES = ("src", "scripts", "tests")


def python_files() -> list[Path]:
    files: list[Path] = []
    for directory in SOURCE_DIRECTORIES:
        files.extend(sorted((REPO_ROOT / directory).rglob("*.py")))
    return [f for f in files if "__pycache__" not in f.parts]


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


@pytest.mark.parametrize("path", python_files(), ids=relative)
def test_file_parses(path: Path):
    """A stray null byte from an editor once made a module unimportable."""
    source = path.read_bytes()
    assert b"\x00" not in source, f"{relative(path)} contains a null byte"
    ast.parse(source.decode("utf-8"), filename=str(path))


@pytest.mark.parametrize("path", python_files(), ids=relative)
def test_text_file_access_declares_an_encoding(path: Path):
    """The prototype wrote its report with a locale dependent encoding.

    On Windows that means cp1252, so a paper title carrying an accented character
    is silently mangled or raises at write time. Every text mode open in this
    repository states its encoding.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function = node.func
        is_builtin_open = isinstance(function, ast.Name) and function.id == "open"
        is_path_open = isinstance(function, ast.Attribute) and function.attr == "open"
        if not (is_builtin_open or is_path_open):
            continue

        # Image.open reads bytes and takes no encoding. It is named here rather
        # than matched loosely so that a genuine text open cannot hide behind it.
        if (is_path_open and isinstance(function.value, ast.Name)
                and function.value.id == "Image"):
            continue

        keywords = {keyword.arg for keyword in node.keywords}
        if "encoding" in keywords:
            continue

        mode = ""
        positional_mode_index = 1 if is_builtin_open else 0
        if len(node.args) > positional_mode_index:
            argument = node.args[positional_mode_index]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                mode = argument.value
        for keyword in node.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                mode = str(keyword.value.value)

        if "b" in mode:
            continue

        offenders.append(node.lineno)

    assert not offenders, (
        f"{relative(path)} opens a text file without encoding= at line(s) "
        f"{', '.join(str(line) for line in offenders)}"
    )


@pytest.mark.parametrize("path", python_files(), ids=relative)
def test_seed_is_imported_not_redefined(path: Path):
    """One seed for the project, defined once in cdfm.__init__.

    A module that sets its own random_state can drift from the rest of the
    project without any test failing, which quietly breaks the claim that two
    models were compared under identical conditions.
    """
    if path.name == "__init__.py" and path.parent.name == "cdfm":
        return

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "RANDOM_STATE":
                pytest.fail(
                    f"{relative(path)} line {node.lineno} redefines RANDOM_STATE. "
                    "Import it from cdfm instead."
                )
