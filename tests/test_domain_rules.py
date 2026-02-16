from __future__ import annotations

import unittest

import main


class TestDomainRules(unittest.TestCase):
    def test_last_match_wins_with_allow_and_deny_conflicts(self) -> None:
        compiled = main.compile_domain_rules(
            [
                "-*.google.com",
                "+mail.google.com",
                "-mail.google.com",
                "+accounts.google.com",
            ]
        )

        mail = main.resolve_domain_policy("mail.google.com", compiled)
        accounts = main.resolve_domain_policy("accounts.google.com", compiled)
        docs = main.resolve_domain_policy("docs.google.com", compiled)
        nomatch = main.resolve_domain_policy("example.org", compiled)

        self.assertFalse(mail.allowed)
        self.assertIsNotNone(mail.winning_rule)
        self.assertEqual(mail.winning_rule.pattern, "mail.google.com")
        self.assertEqual(mail.winning_rule.action, "deny")

        self.assertTrue(accounts.allowed)
        self.assertIsNotNone(accounts.winning_rule)
        self.assertEqual(accounts.winning_rule.pattern, "accounts.google.com")
        self.assertEqual(accounts.winning_rule.action, "allow")

        self.assertFalse(docs.allowed)
        self.assertIsNotNone(docs.winning_rule)
        self.assertEqual(docs.winning_rule.pattern, "*.google.com")
        self.assertEqual(docs.winning_rule.action, "deny")

        self.assertTrue(nomatch.allowed)
        self.assertIsNone(nomatch.winning_rule)

    def test_glob_and_case_insensitive_host_matching(self) -> None:
        compiled = main.compile_domain_rules(
            [
                "-*.bank.*",
                "+safe.bank.com",
            ]
        )

        denied = main.resolve_domain_policy("LOGIN.BANK.CO.UK", compiled)
        allowed = main.resolve_domain_policy("SAFE.BANK.COM", compiled)

        self.assertFalse(denied.allowed)
        self.assertTrue(allowed.allowed)

    def test_invalid_rule_format_raises(self) -> None:
        with self.assertRaises(ValueError):
            main.compile_domain_rules(["gmail.com"])

    def test_invalid_empty_pattern_raises(self) -> None:
        with self.assertRaises(ValueError):
            main.compile_domain_rules(["-"])

    def test_exact_domain_rule_does_not_match_subdomain(self) -> None:
        compiled = main.compile_domain_rules(["-gmail.com"])
        apex = main.resolve_domain_policy("gmail.com", compiled)
        subdomain = main.resolve_domain_policy("mail.gmail.com", compiled)

        self.assertFalse(apex.allowed)
        self.assertTrue(subdomain.allowed)

    def test_subdomain_wildcard_does_not_match_apex(self) -> None:
        compiled = main.compile_domain_rules(["-*.gmail.com"])
        apex = main.resolve_domain_policy("gmail.com", compiled)
        subdomain = main.resolve_domain_policy("mail.gmail.com", compiled)

        self.assertTrue(apex.allowed)
        self.assertFalse(subdomain.allowed)

    def test_catch_all_deny_then_specific_allows(self) -> None:
        compiled = main.compile_domain_rules(
            [
                "-*",
                "+github.com",
                "+*.github.com",
            ]
        )
        github = main.resolve_domain_policy("github.com", compiled)
        gist = main.resolve_domain_policy("gist.github.com", compiled)
        reddit = main.resolve_domain_policy("reddit.com", compiled)

        self.assertTrue(github.allowed)
        self.assertTrue(gist.allowed)
        self.assertFalse(reddit.allowed)

    def test_resolve_domain_policy_accepts_url_with_subpath_query_and_port(self) -> None:
        compiled = main.compile_domain_rules(["-mail.google.com"])
        decision = main.resolve_domain_policy(
            "https://MAIL.GOOGLE.COM:443/mail/u/0/?foo=bar#inbox",
            compiled,
        )

        self.assertEqual(decision.host, "mail.google.com")
        self.assertFalse(decision.allowed)

    def test_matched_rules_trace_is_in_order(self) -> None:
        compiled = main.compile_domain_rules(
            [
                "-*.google.com",
                "+mail.google.com",
                "-mail.google.com",
            ]
        )

        decision = main.resolve_domain_policy("mail.google.com", compiled)
        self.assertEqual(
            [rule.raw for rule in decision.matched_rules],
            ["-*.google.com", "+mail.google.com", "-mail.google.com"],
        )
        self.assertIsNotNone(decision.winning_rule)
        self.assertEqual(decision.winning_rule.raw, "-mail.google.com")

    def test_rule_with_path_blocks_only_matching_path(self) -> None:
        compiled = main.compile_domain_rules(["-mail.google.com/mail/*"])
        denied = main.resolve_domain_policy("https://mail.google.com/mail/u/0/#inbox", compiled)
        allowed_other_path = main.resolve_domain_policy("https://mail.google.com/calendar", compiled)

        self.assertFalse(denied.allowed)
        self.assertTrue(allowed_other_path.allowed)

    def test_path_rules_still_follow_last_match_wins(self) -> None:
        compiled = main.compile_domain_rules(
            [
                "-mail.google.com/mail/*",
                "+mail.google.com/mail/allowed/*",
            ]
        )
        denied = main.resolve_domain_policy("https://mail.google.com/mail/u/0", compiled)
        allowed = main.resolve_domain_policy("https://mail.google.com/mail/allowed/x", compiled)

        self.assertFalse(denied.allowed)
        self.assertTrue(allowed.allowed)
        self.assertIsNotNone(allowed.winning_rule)
        self.assertEqual(allowed.winning_rule.raw, "+mail.google.com/mail/allowed/*")

    def test_invalid_pattern_with_whitespace_raises(self) -> None:
        with self.assertRaises(ValueError):
            main.compile_domain_rules(["-mail .google.com"])

    def test_set_config_key_domain_rules_parses_csv_and_validates(self) -> None:
        config = main.AppConfig()
        main.set_config_key(
            config,
            "ingest.domain_rules",
            "-*.google.com,+mail.google.com,-gmail.com",
        )
        self.assertEqual(
            config.ingest.domain_rules,
            ["-*.google.com", "+mail.google.com", "-gmail.com"],
        )

    def test_set_config_key_domain_rules_rejects_invalid(self) -> None:
        config = main.AppConfig()
        with self.assertRaises(ValueError):
            main.set_config_key(config, "ingest.domain_rules", "gmail.com")
