# Contributing to meta-encoder-eval

We want to make contributing to this project as easy and transparent as possible.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've changed evaluation behavior, rerun the affected suites and check them with
   `python compare.py results/ --per-task`.
3. If you've changed setup, data preparation or the protocol, update the README.
4. Make sure every new source file starts with the copyright header used in this repo.
5. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need to do this once
to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has
sufficient instructions to be able to reproduce the issue: the command you ran, the suite and
task, your GPU setup and package versions, and the scores you got.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe disclosure of security
bugs. In those cases, please go through the process outlined on that page and do not file a
public issue.

## Code of Conduct

This project follows Meta's [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing to meta-encoder-eval, you agree that your contributions will be licensed under
the LICENSE file in the root directory of this source tree.
