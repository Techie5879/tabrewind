from __future__ import annotations

import argparse
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

import main


class TestDomainsCli(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.config_path = self.root / "config.toml"
        config = main.AppConfig()
        config.ingest.domain_rules = ["-*.gmail.com", "+mail.google.com", "-github.com"]
        main.save_config(self.config_path, config)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _run_domains_command(self, domains_command: str, **kwargs: object) -> str:
        args = argparse.Namespace(
            config=self.config_path,
            command="domains",
            domains_command=domains_command,
            allow=None,
            deny=None,
            index=None,
            from_index=None,
            to_index=None,
            host=None,
            since_days=2.0,
            browsers=None,
        )
        for key, value in kwargs.items():
            setattr(args, key, value)

        stream = io.StringIO()
        console = Console(file=stream, force_terminal=True, color_system="standard")
        exit_code = main.run_domains_command(args, console)
        self.assertEqual(exit_code, 0)
        return stream.getvalue()

    @staticmethod
    def _plain(text: str) -> str:
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def test_domains_check_reports_deny_with_winning_rule_and_ansi(self) -> None:
        output = self._run_domains_command("check", host="https://accounts.gmail.com/inbox")
        plain = self._plain(output)
        self.assertIn("DENY accounts.gmail.com", plain)
        self.assertIn("Winning rule: #1", plain)
        self.assertIn("*.gmail.com", plain)
        self.assertIn("\x1b[", output)

    def test_domains_check_reports_allow_override(self) -> None:
        output = self._run_domains_command("check", host="mail.google.com")
        plain = self._plain(output)
        self.assertIn("ALLOW mail.google.com", plain)
        self.assertIn("Winning rule: #2", plain)
        self.assertIn("mail.google.com", plain)

    def test_domains_preview_shows_resolved_view_and_counts(self) -> None:
        with patch(
            "main._collect_recent_hosts",
            return_value=(["accounts.gmail.com", "github.com", "mail.google.com", "x.com"], []),
        ):
            output = self._run_domains_command("preview", since_days=7.0, browsers="zen")
        plain = self._plain(output)

        self.assertIn("DENY accounts.gmail.com", plain)
        self.assertIn("DENY github.com", plain)
        self.assertIn("ALLOW mail.google.com", plain)
        self.assertIn("ALLOW x.com", plain)
        self.assertIn("Resolved hosts: total=4", plain)
        self.assertIn("allowed=2", plain)
        self.assertIn("denied=2", plain)

    def test_domains_add_insert_move_remove_persists_order(self) -> None:
        self._run_domains_command("add", deny="*.bank.*")
        self._run_domains_command("insert", index=2, allow="safe.bank.com")
        self._run_domains_command("move", from_index=5, to_index=1)
        self._run_domains_command("remove", index=5)

        config = main.load_config(self.config_path)
        self.assertEqual(
            config.ingest.domain_rules,
            ["-*.bank.*", "-*.gmail.com", "+safe.bank.com", "+mail.google.com"],
        )
