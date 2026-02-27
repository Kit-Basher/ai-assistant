#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


METHOD_ORDER = ("GET", "POST", "PUT", "DELETE")


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _const_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return None


def _parts_index(node: ast.AST) -> int | None:
    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != "parts":
        return None
    index_node = node.slice
    if isinstance(index_node, ast.Constant) and isinstance(index_node.value, int):
        return int(index_node.value)
    return None


def _flatten_and(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        output: list[ast.AST] = []
        for value in node.values:
            output.extend(_flatten_and(value))
        return output
    return [node]


def _extract_path_in_values(test: ast.AST) -> list[str]:
    if not isinstance(test, ast.Compare):
        return []
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return []
    if not isinstance(test.left, ast.Name) or test.left.id != "path":
        return []
    if not isinstance(test.ops[0], ast.In):
        return []
    comparator = test.comparators[0]
    if not isinstance(comparator, (ast.Set, ast.List, ast.Tuple)):
        return []
    values = [_const_str(item) for item in comparator.elts]
    return [item for item in values if isinstance(item, str) and item.startswith("/")]


def _extract_path_eq_value(test: ast.AST) -> str | None:
    if not isinstance(test, ast.Compare):
        return None
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return None
    if not isinstance(test.ops[0], ast.Eq):
        return None
    if isinstance(test.left, ast.Name) and test.left.id == "path":
        return _const_str(test.comparators[0])
    if isinstance(test.comparators[0], ast.Name) and test.comparators[0].id == "path":
        return _const_str(test.left)
    return None


def _extract_parts_route(test: ast.AST) -> str | None:
    parts = _flatten_and(test)
    length: int | None = None
    fixed_parts: dict[int, str] = {}

    for item in parts:
        if not isinstance(item, ast.Compare):
            continue
        if len(item.ops) != 1 or len(item.comparators) != 1:
            continue
        op = item.ops[0]
        comparator = item.comparators[0]

        # len(parts) == N
        if isinstance(op, ast.Eq) and isinstance(item.left, ast.Call):
            call = item.left
            if (
                isinstance(call.func, ast.Name)
                and call.func.id == "len"
                and len(call.args) == 1
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "parts"
            ):
                value = _const_int(comparator)
                if value is not None and value >= 1:
                    length = value
            continue

        if not isinstance(op, ast.Eq):
            continue

        # parts[i] == "literal"
        left_index = _parts_index(item.left)
        left_value = _const_str(comparator)
        if left_index is not None and left_value is not None:
            fixed_parts[left_index] = left_value
            continue

        # "literal" == parts[i]
        right_index = _parts_index(comparator)
        right_value = _const_str(item.left)
        if right_index is not None and right_value is not None:
            fixed_parts[right_index] = right_value
            continue

    if length is None or not fixed_parts:
        return None
    segments = [f"{{part{idx}}}" for idx in range(length)]
    for idx, value in fixed_parts.items():
        if 0 <= idx < length:
            segments[idx] = value
    return "/" + "/".join(segments)


def _extract_routes_from_test(test: ast.AST) -> list[str]:
    routes: list[str] = []
    direct = _extract_path_eq_value(test)
    if direct and direct.startswith("/"):
        routes.append(direct)
    routes.extend(_extract_path_in_values(test))
    parts_route = _extract_parts_route(test)
    if parts_route:
        routes.append(parts_route)
    return routes


def _iter_child_statement_lists(node: ast.stmt) -> Iterable[list[ast.stmt]]:
    if isinstance(node, ast.If):
        yield node.body
        yield node.orelse
        return
    if isinstance(node, ast.For):
        yield node.body
        yield node.orelse
        return
    if isinstance(node, ast.While):
        yield node.body
        yield node.orelse
        return
    if isinstance(node, ast.With):
        yield node.body
        return
    if isinstance(node, ast.Try):
        yield node.body
        for handler in node.handlers:
            yield handler.body
        yield node.orelse
        yield node.finalbody
        return
    if isinstance(node, ast.Match):
        for case in node.cases:
            yield case.body


def _extract_from_statements(statements: list[ast.stmt], *, bucket: set[str]) -> None:
    for node in statements:
        if isinstance(node, ast.If):
            for route in _extract_routes_from_test(node.test):
                bucket.add(route)
        for child_list in _iter_child_statement_lists(node):
            _extract_from_statements(child_list, bucket=bucket)


def extract_routes_from_api_server(api_server_path: Path) -> dict[str, list[str]]:
    module = ast.parse(api_server_path.read_text(encoding="utf-8"))
    output: dict[str, set[str]] = {method: set() for method in METHOD_ORDER}

    for top in module.body:
        if not isinstance(top, ast.ClassDef) or top.name != "APIServerHandler":
            continue
        for item in top.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name not in {"do_GET", "do_POST", "do_PUT", "do_DELETE"}:
                continue
            method = item.name.replace("do_", "")
            if method not in output:
                continue
            _extract_from_statements(item.body, bucket=output[method])

    return {method: sorted(output[method]) for method in METHOD_ORDER}


def format_routes_markdown(routes_by_method: dict[str, list[str]]) -> str:
    lines = ["## Active Endpoints"]
    for method in METHOD_ORDER:
        lines.append(f"### {method}")
        for route in routes_by_method.get(method, []):
            lines.append(f"- {route}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    api_server_path = repo_root / "agent" / "api_server.py"
    routes = extract_routes_from_api_server(api_server_path)
    print(format_routes_markdown(routes), end="")


if __name__ == "__main__":
    main()
