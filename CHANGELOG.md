# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Moved `matplotlib` from core dependencies into the optional `plot` extra (`qoraphase[plot]`) for a lighter core install. Plain installs no longer pull `matplotlib`, while `Reconstruction.plot()` raises `ImportError` naming `qoraphase[plot]` when `matplotlib` is absent.

## [0.2.0] - 2026-10-21

Added radially averaged PRTF (`prtf`, `prtf_curve`), sub-pixel DFT registration, linear phase-ramp removal, partial coherence and phase-constraint stages, and the generational `genetic` algorithm.

## [0.1.0] - 2026-10-15

Initial PyPI release with composable phase retrieval (`ER`, `HIO`, `RAAR`, `Shrinkwrap`, `Ensemble`, `postprocess`, `detwin`, and `diagnose`).
