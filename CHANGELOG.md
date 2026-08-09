# Changelog

All notable changes to gs-engine are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Moved shifted weak-Popov row reduction to `gfm::weak_popov`, retaining the
existing polynomial representation, deterministic leading-term order, and
interpolation results.
- Replaced the former `fff` dependency with `fgf` and updated `cafft` to the
shared field types and dispatch backend.
