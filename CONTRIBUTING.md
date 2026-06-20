# Contributing

## Branches

Branch names follow `<type>/<short-description>`, where `<type>` is one of
`docs`, `feature`, `bug`, `fix`, etc. (e.g. `feature/new-sign-in`).

## Local checks (match CI before pushing)

CI (`.github/workflows/ci.yml`) runs on the lightweight `requirements-dev.txt`
with network clients and FinBERT mocked. Run the same gates locally:

```bash
black --check src tests scripts
isort --check-only src tests scripts
mypy src
pytest tests -q
```

(`black src tests scripts && isort src tests scripts` to autoformat.)

## Merge policy

**Pull requests are merged with "Rebase and merge" — this is the proper/standard
merge method for this repository.** Do not squash and do not create merge commits.

Rationale: rebasing keeps `main` a linear history of the individual, already-tested
commits, which keeps `git bisect`, `git log`, and `git blame` clean and makes each
change easy to revert in isolation.

Before merging:

- CI must be green and the PR must be ready for review (not a draft).
- Keep the branch's commit history tidy (squash fixups locally) so the linear
  history that lands on `main` stays meaningful.
