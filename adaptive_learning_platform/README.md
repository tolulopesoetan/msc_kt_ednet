# SFLA Adaptive Learning Platform

## Overview

This application is a proof-of-concept adaptive learning platform for the University of the West of England module Sets, Functions and Linear Algebra (SFLA).

The platform uses Bayesian Knowledge Tracing (BKT) to maintain a separate mastery estimate for each of nine knowledge components. After every response, the relevant mastery estimate is updated and the recommendation engine selects another question according to the learner's estimated weaknesses.

The application forms the implementation component of a wider dissertation comparing BKT, Deep Knowledge Tracing (DKT) and causal Self-Attentive Knowledge Tracing (SAKT).

## Research scope

The platform demonstrates:

- Integration of a knowledge-tracing model into an interactive learning application
- Skill-specific mastery estimation
- Adaptive question selection
- Difficulty selection based on estimated mastery
- Immediate answer feedback
- Learner-progress visualisation
- Persistent interaction logging
- Research-data export
- Automated and functional testing

The SFLA BKT parameters were fitted to simulated learner interactions. They are used to demonstrate technical execution only. The platform has not been evaluated using real SFLA learner-response sequences and does not establish improved learning outcomes.

## Knowledge components

The application represents nine SFLA knowledge components:

| Skill | Knowledge component |
|---|---|
| KC01 | Logical equivalence and truth tables |
| KC02 | Quantifiers and statement transformations |
| KC03 | Set notation and membership |
| KC04 | Set operations |
| KC05 | Power sets and Cartesian products |
| KC06 | Set-based proof |
| KC07 | Mathematical induction |
| KC08 | Function properties |
| KC09 | Inverse and composite functions |

## Architecture

```mermaid
flowchart TD
    UI["Streamlit interface"] --> Selector["Adaptive question selector"]
    Selector --> Bank["SFLA question bank"]
    UI --> BKT["Per-skill BKT engine"]
    BKT --> Params["Exported BKT parameters"]
    UI --> Database["SQLite interaction store"]
    Database --> Dashboard["Progress dashboard"]
    Database --> Export["Research export"]