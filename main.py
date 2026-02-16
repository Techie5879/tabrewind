from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from tabrewind_core.app_types import AppConfig, EncodedEntryPreview, HistoryEntry, IngestSummary, ProfileStore
from tabrewind_core.config_store import (
    CONFIG_FILE_NAME,
    SUPPORTED_BROWSERS,
    init_config,
    load_config,
    parse_csv_list,
    render_config_toml,
    save_config,
    set_config_key,
)
from tabrewind_core.domain_policy import (
    DomainPolicyDecision,
    DomainRule,
    compile_domain_rules,
    format_domain_rule,
    resolve_domain_policy,
)
from tabrewind_core.history_ops import (
    build_embedding_input,
    chunk_entries,
    dedupe_entries,
    discover_places_files,
    embed_batch,
    load_history_entries,
    normalize_url,
    run_ingest,
)
from tabrewind_core.ingest_db_ops import collect_recent_hosts as _collect_recent_hosts
from tabrewind_core.ingest_db_ops import run_ingest_db
from tabrewind_core.vectorization import VectorizationClient, fetch_server_slots


INGEST_DB_FILE_NAME = "tabrewind.sqlite"
_format_domain_rule = format_domain_rule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tabrewind",
        description="Config-first CLI for local browser history ingestion.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CONFIG_FILE_NAME),
        help="Path to config TOML file (default: ./config.toml).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create default config.toml.")
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config.toml.",
    )

    config_cmd = subparsers.add_parser("config", help="Inspect or update config values.")
    config_subparsers = config_cmd.add_subparsers(dest="config_command", required=True)
    config_subparsers.add_parser("show", help="Print resolved config.")
    config_subparsers.add_parser("path", help="Print config path.")

    config_set = config_subparsers.add_parser("set", help="Set a config key and save.")
    config_set.add_argument("key", help="Dot-path key, e.g. llama.base_url")
    config_set.add_argument("value", help="New value")

    ingest = subparsers.add_parser(
        "ingest",
        help="Load browser history and print embedding vector previews.",
    )
    ingest.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional one-shot override for max deduped items.",
    )

    ingest_db = subparsers.add_parser(
        "ingest-db",
        help="Ingest browser history into local SQLite schema.",
    )
    ingest_db.add_argument(
        "--since-days",
        type=float,
        default=2.0,
        help="Only ingest visits from the last N days (default: 2).",
    )
    ingest_db.add_argument(
        "--browsers",
        type=str,
        default=None,
        help="Optional CSV override for browsers (e.g. zen,firefox).",
    )

    domains = subparsers.add_parser(
        "domains",
        help="Manage host allow/deny glob rules.",
    )
    domains_subparsers = domains.add_subparsers(dest="domains_command", required=True)

    domains_subparsers.add_parser("list", help="List configured domain rules.")

    domains_add = domains_subparsers.add_parser("add", help="Append a new domain rule.")
    add_group = domains_add.add_mutually_exclusive_group(required=True)
    add_group.add_argument("--allow", type=str, help="Allow pattern (without '+' prefix).")
    add_group.add_argument("--deny", type=str, help="Deny pattern (without '-' prefix).")

    domains_insert = domains_subparsers.add_parser("insert", help="Insert a domain rule by order.")
    domains_insert.add_argument("--index", type=int, required=True, help="1-based insertion index.")
    insert_group = domains_insert.add_mutually_exclusive_group(required=True)
    insert_group.add_argument("--allow", type=str, help="Allow pattern (without '+' prefix).")
    insert_group.add_argument("--deny", type=str, help="Deny pattern (without '-' prefix).")

    domains_remove = domains_subparsers.add_parser("remove", help="Remove a domain rule by index.")
    domains_remove.add_argument("--index", type=int, required=True, help="1-based rule index.")

    domains_move = domains_subparsers.add_parser("move", help="Move a domain rule to a new index.")
    domains_move.add_argument("--from-index", type=int, required=True, help="1-based source index.")
    domains_move.add_argument("--to-index", type=int, required=True, help="1-based destination index.")

    domains_check = domains_subparsers.add_parser(
        "check", help="Resolve a single host against configured domain rules."
    )
    domains_check.add_argument("host", type=str, help="Host or URL to evaluate.")

    domains_preview = domains_subparsers.add_parser(
        "preview", help="Preview resolved allow/deny decisions for discovered hosts."
    )
    domains_preview.add_argument(
        "--since-days",
        type=float,
        default=2.0,
        help="Only include hosts with visits from the last N days (default: 2).",
    )
    domains_preview.add_argument(
        "--browsers",
        type=str,
        default=None,
        help="Optional CSV override for browsers (e.g. zen,firefox).",
    )

    return parser.parse_args()


def _render_domain_decision_status(decision: DomainPolicyDecision) -> str:
    if decision.allowed:
        return "[green]ALLOW[/green]"
    return "[red]DENY[/red]"


def run_domains_command(args: argparse.Namespace, console: Console) -> int:
    config_path: Path = args.config
    config = load_config(config_path)

    if args.domains_command == "list":
        compiled = compile_domain_rules(config.ingest.domain_rules)
        if not compiled:
            console.print("No domain rules configured.")
            console.print("Default policy: [green]ALLOW[/green] when no rule matches.")
            return 0

        console.print("Domain rules (top to bottom, last match wins):")
        for rule in compiled:
            status = "[green]+ allow[/green]" if rule.action == "allow" else "[red]- deny[/red]"
            console.print(f"{rule.index}. {status} {rule.pattern}")
        return 0

    if args.domains_command in {"add", "insert"}:
        action = "allow" if getattr(args, "allow", None) is not None else "deny"
        pattern = args.allow if action == "allow" else args.deny
        rule = _format_domain_rule(action, pattern)
        updated = list(config.ingest.domain_rules)
        if args.domains_command == "add":
            updated.append(rule)
        else:
            index = int(args.index)
            if index <= 0:
                raise ValueError("Rule index must be >= 1")
            insert_at = min(index - 1, len(updated))
            updated.insert(insert_at, rule)

        compile_domain_rules(updated)
        config.ingest.domain_rules = updated
        save_config(config_path, config)
        console.print(f"Added rule {rule}")
        return 0

    if args.domains_command == "remove":
        updated = list(config.ingest.domain_rules)
        if not updated:
            raise ValueError("No domain rules configured")
        index = int(args.index)
        if index <= 0 or index > len(updated):
            raise ValueError(f"Rule index out of range: {index}")
        removed = updated.pop(index - 1)
        compile_domain_rules(updated)
        config.ingest.domain_rules = updated
        save_config(config_path, config)
        console.print(f"Removed rule {removed}")
        return 0

    if args.domains_command == "move":
        updated = list(config.ingest.domain_rules)
        if not updated:
            raise ValueError("No domain rules configured")
        from_index = int(args.from_index)
        to_index = int(args.to_index)
        if from_index <= 0 or from_index > len(updated):
            raise ValueError(f"from-index out of range: {from_index}")
        if to_index <= 0:
            raise ValueError("to-index must be >= 1")
        item = updated.pop(from_index - 1)
        destination = min(to_index - 1, len(updated))
        updated.insert(destination, item)
        compile_domain_rules(updated)
        config.ingest.domain_rules = updated
        save_config(config_path, config)
        console.print(f"Moved rule {item} to position {destination + 1}")
        return 0

    compiled = compile_domain_rules(config.ingest.domain_rules)
    if args.domains_command == "check":
        decision = resolve_domain_policy(args.host, compiled)
        status = _render_domain_decision_status(decision)
        console.print(f"{status} {decision.host}")
        if decision.winning_rule is None:
            console.print("Winning rule: (none) -> default allow")
        else:
            winning = decision.winning_rule
            marker = "[green]allow[/green]" if winning.action == "allow" else "[red]deny[/red]"
            console.print(f"Winning rule: #{winning.index} {marker} {winning.pattern}")

        if decision.matched_rules:
            console.print("Matched rules:")
            for rule in decision.matched_rules:
                marker = "[green]allow[/green]" if rule.action == "allow" else "[red]deny[/red]"
                console.print(f"- #{rule.index} {marker} {rule.pattern}")
        return 0

    if args.domains_command == "preview":
        browsers_override: list[str] | None = None
        if args.browsers:
            browsers_override = parse_csv_list(args.browsers)
            invalid = [browser for browser in browsers_override if browser not in SUPPORTED_BROWSERS]
            if invalid:
                raise ValueError(f"Unsupported browser(s): {', '.join(invalid)}")

        hosts, errors = _collect_recent_hosts(
            config,
            since_days=args.since_days,
            browsers_override=browsers_override,
        )
        if not hosts:
            console.print("No hosts discovered for preview window.")
        allowed_count = 0
        denied_count = 0
        for host in hosts:
            decision = resolve_domain_policy(host, compiled)
            status = _render_domain_decision_status(decision)
            if decision.allowed:
                allowed_count += 1
            else:
                denied_count += 1
            if decision.winning_rule is None:
                console.print(f"{status} {host} (default)")
            else:
                marker = "allow" if decision.winning_rule.action == "allow" else "deny"
                console.print(
                    f"{status} {host} (rule #{decision.winning_rule.index}: {marker} {decision.winning_rule.pattern})"
                )

        console.print(
            f"Resolved hosts: total={len(hosts)} [green]allowed={allowed_count}[/green] [red]denied={denied_count}[/red]"
        )
        if errors:
            console.print("Profile read errors:")
            for message in errors:
                console.print(f"- {message}")
        return 0

    raise RuntimeError(f"Unknown domains command: {args.domains_command}")


def run_init_command(args: argparse.Namespace, console: Console) -> int:
    config_path: Path = args.config
    _, did_write = init_config(config_path, force=args.force)
    if did_write:
        console.print(f"Wrote config to {config_path}")
    else:
        console.print(f"Config already exists at {config_path}. Use --force to overwrite.")
    return 0


def run_config_command(args: argparse.Namespace, console: Console) -> int:
    config_path: Path = args.config
    if args.config_command == "path":
        console.print(str(config_path))
        return 0

    config = load_config(config_path)
    if args.config_command == "show":
        console.print(render_config_toml(config), markup=False)
        return 0

    if args.config_command == "set":
        set_config_key(config, args.key, args.value)
        save_config(config_path, config)
        console.print(f"Updated {args.key} in {config_path}")
        return 0

    raise RuntimeError(f"Unknown config command: {args.config_command}")


def run_ingest_command(args: argparse.Namespace, console: Console) -> int:
    config = load_config(args.config)
    if args.max_items is not None:
        config.ingest.max_items = None if args.max_items <= 0 else args.max_items

    summary = run_ingest(config)
    if summary.discovered_profile_count == 0:
        console.print("No places.sqlite files were discovered for configured browsers.")
        return 0

    console.print(
        "Discovered "
        f"{summary.discovered_profile_count} profile DB(s); loaded {summary.loaded_row_count} rows; "
        f"deduped to {summary.deduped_row_count} rows; embedding workers {summary.effective_embedding_workers}."
    )

    if summary.load_errors:
        console.print("Profile read errors:")
        for message in summary.load_errors:
            console.print(f"- {message}")

    if summary.embed_errors:
        console.print("Embedding errors:")
        for message in summary.embed_errors:
            console.print(f"- {message}")

    for preview in summary.previews:
        vector_values = ", ".join(f"{value:.6f}" for value in preview.vector_preview)
        console.print(
            f"[{preview.entry.browser}:{preview.entry.profile}] {preview.entry.title} -> {preview.entry.canonical_url}"
        )
        if preview.llm_tags:
            console.print(f"tags: {', '.join(preview.llm_tags)}")
        console.print(f"vec[:{len(preview.vector_preview)}]: [{vector_values}]")

    return 0


def run_ingest_db_command(args: argparse.Namespace, console: Console) -> int:
    config = load_config(args.config)
    db_path = Path(INGEST_DB_FILE_NAME)

    browsers_override: list[str] | None = None
    if args.browsers:
        browsers_override = parse_csv_list(args.browsers)
        invalid = [browser for browser in browsers_override if browser not in SUPPORTED_BROWSERS]
        if invalid:
            raise ValueError(f"Unsupported browser(s): {', '.join(invalid)}")

    result = run_ingest_db(
        config,
        db_path=db_path,
        since_days=args.since_days,
        browsers_override=browsers_override,
    )

    console.print(
        "DB ingest complete: "
        f"profiles={result['discovered_profiles']} "
        f"source_rows={result['source_rows']} "
        f"allowed_rows={result['filtered_allowed_rows']} "
        f"denied_rows={result['filtered_denied_rows']} "
        f"pages_touched={result['pages_touched']} "
        f"visits_inserted={result['visits_inserted']} "
        f"fts_synced={result['fts_rows_synced']} "
        f"embeddings_synced={result['embeddings_synced']}"
    )
    console.print(f"DB path: {result['db_path']}")

    load_errors = result["load_errors"]
    if isinstance(load_errors, list) and load_errors:
        console.print("Profile read errors:")
        for message in load_errors:
            console.print(f"- {message}")

    embed_errors = result["embed_errors"]
    if isinstance(embed_errors, list) and embed_errors:
        console.print("Embedding sync errors:")
        for message in embed_errors:
            console.print(f"- {message}")

    return 0


def main() -> int:
    args = parse_args()
    console = Console(stderr=False)

    if args.command == "init":
        return run_init_command(args, console)
    if args.command == "config":
        return run_config_command(args, console)
    if args.command == "ingest":
        return run_ingest_command(args, console)
    if args.command == "ingest-db":
        return run_ingest_db_command(args, console)
    if args.command == "domains":
        return run_domains_command(args, console)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
