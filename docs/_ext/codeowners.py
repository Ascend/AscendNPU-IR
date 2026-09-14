"""Render the repository CODEOWNERS file as a searchable dashboard."""

from collections import OrderedDict
from dataclasses import dataclass
from html import escape
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive


DEFAULT_REVIEWERS = {"@hid90099092", "@tanshengshun"}


@dataclass(frozen=True)
class Rule:
    pattern: str
    owners: tuple[str, ...]


def parse_codeowners(path: Path) -> OrderedDict[str, list[Rule]]:
    """Group CODEOWNERS rules under their nearest section comment."""
    groups: OrderedDict[str, list[Rule]] = OrderedDict()
    section = "Other"

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line[1:].strip()
            if heading.endswith(".") and not heading.startswith(
                ("This file", "For each file")
            ):
                section = heading[:-1]
            continue

        fields = line.split()
        if len(fields) < 2:
            continue
        groups.setdefault(section, []).append(Rule(fields[0], tuple(fields[1:])))

    return groups


def render_dashboard(groups: OrderedDict[str, list[Rule]]) -> str:
    """Return accessible dashboard markup for parsed CODEOWNERS groups."""
    rules = [rule for group in groups.values() for rule in group]
    owners = sorted(
        {
            owner
            for rule in rules
            for owner in rule.owners
            if owner not in DEFAULT_REVIEWERS
        },
        key=str.lower,
    )
    cards = []

    for group, group_rules in groups.items():
        group_owners = sorted(
            {
                owner
                for rule in group_rules
                for owner in rule.owners
                if owner not in DEFAULT_REVIEWERS
            },
            key=str.lower,
        )
        owner_chips = "".join(
            f'<a class="co-owner" href="https://gitcode.com/{escape(owner[1:])}" '
            f'target="_blank" rel="noopener noreferrer">{escape(owner)}</a>'
            for owner in group_owners
        )
        rows = []
        for rule in group_rules:
            rule_owners = [
                owner for owner in rule.owners if owner not in DEFAULT_REVIEWERS
            ]
            rendered_owners = " ".join(
                f'<span class="co-owner-inline">{escape(owner)}</span>'
                for owner in rule_owners
            )
            if not rendered_owners:
                rendered_owners = '<span class="co-unassigned">No module owner</span>'
            search_text = " ".join((group, rule.pattern, *rule_owners)).lower()
            rows.append(
                '<tr class="co-rule" data-search="{}">'
                '<td><code>{}</code></td><td>{}</td></tr>'.format(
                    escape(search_text, quote=True),
                    escape(rule.pattern),
                    rendered_owners,
                )
            )
        rows = "".join(rows)
        group_search = escape(
            (group + " " + " ".join(group_owners)).lower(), quote=True
        )
        cards.append(
            f'<section class="co-card" data-search="{group_search}">'
            f'<div class="co-card-head"><div><h2>{escape(group)}</h2>'
            f'<span class="co-count">{len(group_rules)} rules</span></div>'
            f'<div class="co-owners">{owner_chips}</div></div>'
            '<div class="co-table-wrap"><table><thead><tr><th>Path pattern</th>'
            f'<th>Owners</th></tr></thead><tbody>{rows}</tbody></table></div></section>'
        )

    return (
        '<div class="codeowners-dashboard">'
        '<div class="co-summary">'
        f'<div><strong>{len(groups)}</strong><span>modules</span></div>'
        f'<div><strong>{len(rules)}</strong><span>rules</span></div>'
        f'<div><strong>{len(owners)}</strong><span>owners</span></div>'
        '</div>'
        '<label class="co-search-label" for="codeowners-search">'
        'Search modules, paths, or owners</label>'
        '<input id="codeowners-search" class="co-search" type="search" '
        'placeholder="For example: PlanMemory or @username" autocomplete="off">'
        '<p id="codeowners-empty" class="co-empty" hidden>No matching ownership rules.</p>'
        f'<div id="codeowners-groups">{"".join(cards)}</div></div>'
    )


class CodeOwnersDirective(Directive):
    has_content = False

    def run(self):
        repo_root = Path(self.state.document.settings.env.app.confdir).parent
        source = repo_root / "CODEOWNERS"
        if not source.is_file():
            raise self.error(f"CODEOWNERS not found at {source}")
        self.state.document.settings.env.note_dependency(str(source))
        return [nodes.raw("", render_dashboard(parse_codeowners(source)), format="html")]


def setup(app):
    app.add_directive("codeowners-dashboard", CodeOwnersDirective)
    return {"version": "1.0", "parallel_read_safe": True}
