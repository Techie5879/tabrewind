from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class DomainRule:
    index: int
    raw: str
    action: str
    pattern: str


@dataclass(frozen=True)
class DomainPolicyDecision:
    host: str
    allowed: bool
    winning_rule: DomainRule | None
    matched_rules: list[DomainRule]


def normalize_rule_host_input(value: str) -> str:
    host, _ = _normalize_host_and_path_input(value)
    return host


def _normalize_host_and_path_input(value: str) -> tuple[str, str]:
    candidate = value.strip().lower()
    if not candidate:
        return "", "/"

    if "://" in candidate:
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            return "", "/"
        host = (parsed.hostname or "").lower().rstrip(".")
        path = parsed.path or "/"
        return host, path

    stripped = candidate.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]
    host_part = stripped
    path = "/"
    if "/" in stripped:
        host_part, raw_path = stripped.split("/", maxsplit=1)
        path = "/" + raw_path

    host = host_part.strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if ":" in host and host.count(":") == 1:
        maybe_host, maybe_port = host.rsplit(":", maxsplit=1)
        if maybe_port.isdigit():
            host = maybe_host
    host = host.rstrip(".")
    if not path:
        path = "/"
    return host, path


def _normalize_rule_pattern_input(value: str) -> str:
    candidate = value.strip().lower()
    if not candidate:
        return ""

    if "://" in candidate:
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            return ""
        host = (parsed.hostname or "").lower().rstrip(".")
        if not host:
            return ""
        path = parsed.path or ""
        if path:
            return f"{host}{path}"
        return host

    stripped = candidate.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]
    if "/" in stripped:
        host_part, raw_path = stripped.split("/", maxsplit=1)
        host = normalize_rule_host_input(host_part)
        if not host:
            return ""
        return f"{host}/{raw_path}"

    return normalize_rule_host_input(stripped)


def _domain_rule_target(host: str, path: str, rule: DomainRule) -> str:
    if "/" in rule.pattern:
        return f"{host}{path}"
    else:
        return host


def compile_domain_rules(raw_rules: list[str]) -> list[DomainRule]:
    compiled: list[DomainRule] = []
    for idx, raw in enumerate(raw_rules, start=1):
        text = raw.strip()
        if not text:
            continue
        marker = text[0]
        if marker not in {"+", "-"}:
            raise ValueError(
                f"Invalid domain rule at index {idx}: '{raw}'. "
                "Rules must start with '+' (allow) or '-' (deny)."
            )

        pattern = _normalize_rule_pattern_input(text[1:])
        if not pattern:
            raise ValueError(f"Invalid domain rule at index {idx}: empty pattern")
        if any(char.isspace() for char in pattern):
            raise ValueError(
                f"Invalid domain rule at index {idx}: pattern cannot include whitespace"
            )

        action = "allow" if marker == "+" else "deny"
        compiled.append(DomainRule(index=idx, raw=text, action=action, pattern=pattern))
    return compiled


def resolve_domain_policy(host_or_url: str, compiled_rules: list[DomainRule]) -> DomainPolicyDecision:
    host, path = _normalize_host_and_path_input(host_or_url)
    matched_rules = [
        rule
        for rule in compiled_rules
        if fnmatch.fnmatchcase(_domain_rule_target(host, path, rule), rule.pattern)
    ]
    winning_rule = matched_rules[-1] if matched_rules else None
    allowed = True
    if winning_rule is not None:
        allowed = winning_rule.action == "allow"
    return DomainPolicyDecision(
        host=host,
        allowed=allowed,
        winning_rule=winning_rule,
        matched_rules=matched_rules,
    )


def format_domain_rule(action: str, pattern: str) -> str:
    normalized_pattern = _normalize_rule_pattern_input(pattern)
    if not normalized_pattern:
        raise ValueError("Domain pattern cannot be empty")
    if action not in {"allow", "deny"}:
        raise ValueError(f"Unsupported domain rule action: {action}")
    prefix = "+" if action == "allow" else "-"
    return f"{prefix}{normalized_pattern}"


__all__ = [
    "DomainPolicyDecision",
    "DomainRule",
    "compile_domain_rules",
    "format_domain_rule",
    "resolve_domain_policy",
]
