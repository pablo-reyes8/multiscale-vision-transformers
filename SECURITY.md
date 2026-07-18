# Security policy

## Supported versions

Security fixes are applied to the latest release and the `main` branch.

| Version | Supported |
| --- | --- |
| Latest release | Yes |
| `main` | Yes |
| Older releases | Best effort |

## Reporting a vulnerability

Please do not disclose suspected vulnerabilities in a public issue.

Use GitHub's private vulnerability reporting for this repository. If that option
is unavailable, email `alejogranados229@gmail.com` with:

- the affected version or commit;
- reproduction steps or a proof of concept;
- the expected security impact;
- any suggested mitigation.

You should receive an acknowledgement within seven days. We will coordinate a
fix and disclosure timeline after reproducing and assessing the report.

## Scope

Reports about unsafe checkpoint loading, dependency vulnerabilities, arbitrary
file access, command execution, secrets in automation and container privilege
issues are in scope. General model accuracy or robustness limitations should be
reported as normal bugs unless they create a concrete security impact.

