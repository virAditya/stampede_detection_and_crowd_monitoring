# Multi Camera Crowd Panic and Stampede Detection System
## Overview

This project presents a production ready computer vision system designed for real time detection of crowd panic and stampede risk across multiple camera feeds. Unlike traditional surveillance systems that rely on static people counts or fixed thresholds, this system focuses on understanding crowd behavior. It learns how people normally move within each camera view and raises alerts only when motion patterns shift toward chaos or panic. This approach significantly reduces false alarms while improving early risk detection.

The system is intended for high density public environments such as railway stations, shopping malls, campuses, corridors, and event venues where crowd density alone is not a reliable indicator of danger.

## Problem Statement

Conventional crowd monitoring solutions suffer from fundamental limitations. They apply the same thresholds across all camera views, fail to adapt to local context, and often confuse high density with high risk. As a result, busy areas frequently trigger unnecessary alerts, while early warning signs in smaller or calmer spaces may go unnoticed.

This system addresses these limitations by learning a behavioral baseline independently for each camera. By observing crowd flow over thousands of frames, it builds an understanding of what normal movement looks like in that specific location. Alerts are generated only when statistically meaningful deviations occur.

## Core Concept

Each camera operates as an independent learning unit. During an initial learning phase of approximately five thousand frames, the system models typical motion and density patterns for that view. Once the baseline is established, adaptive anomaly detection is enabled. This ensures that consistently busy areas remain stable, while sudden chaotic motion in normally calm spaces is detected early.

## System Architecture

The architecture is designed for robustness, scalability, and real time performance. Video frames are ingested asynchronously to prevent processing bottlenecks. Crowd density estimation is performed using oriented bounding box detection, while motion behavior is extracted using optical flow analysis. These signals are combined into interpretable risk scores that reflect both density and movement disorder.

A continuous baseline learning module updates adaptive thresholds for each camera. Alerts are triggered only when deviations exceed learned norms rather than predefined static limits. All components run in separate processes using safe inter process communication to avoid deadlocks and ensure long running stability.

## Processing Pipeline

Frames are read asynchronously from video sources. Crowd detection is executed periodically rather than on every frame to reduce computational load. Optical flow analysis is performed at a reduced resolution to capture motion dynamics efficiently. Speed variance and directional entropy are computed and fused with density information to generate risk indicators. These indicators are evaluated against learned baselines to determine whether an alert should be raised.

## Performance Optimizations

The system is optimized for deployment in real environments with multiple concurrent camera feeds. Frame skipping ensures that no camera attempts to process every frame. Optical flow computation is performed at lower resolution without sacrificing behavioral accuracy. Detection and motion analysis are decoupled to prevent pipeline stalls. Asynchronous inference and direct grid computation further reduce overhead.

With these optimizations, the system achieves approximately three to five frames per second per camera while processing high quality video inputs, resulting in a substantial performance improvement over naive implementations.

## Risk Metrics

Rather than relying on a single metric, the system evaluates multiple behavioral indicators. These include crowd density, motion speed variance, directional disorder, chaos indicators, panic signals, and both short term and long term deviations. Each alert is backed by a clear trigger reason, making the system transparent and auditable.

## User Interface

A PyQt based dashboard provides real time visualization and control. Users can monitor live video feeds, observe per camera risk trends, and manage baseline learning independently for each camera. Baseline learning becomes available only after sufficient observation, ensuring stability. All alerts and metrics are logged automatically for later analysis.

## Logging and Analysis

The system maintains detailed CSV logs containing more than fifteen metrics per camera. Each event is timestamped and includes deviation scores and trigger reasons. This logging design supports research evaluation, forensic analysis, and safety audits without additional tooling.

## Project Structure

The repository follows a modular structure that separates camera processing, baseline learning, motion analysis, risk scoring, logging, and user interface components. This organization allows easy experimentation, extension, and integration into larger systems.

## Installation and Usage

Setup is straightforward. Users clone the repository, place the trained detection model in the designated directory, configure input sources, install dependencies, and run the main entry point. The configuration driven design allows rapid testing and customization without modifying core logic.

## Applications

This system is suitable for public transport hubs, campuses, commercial spaces, event venues, and crowd behavior research. Its adaptive design makes it applicable to both academic and commercial deployments.

## Design Philosophy

The system is built on the principle that normal behavior must be learned before anomalies can be detected. Thresholds should adapt to context rather than being hard coded. Perception and decision making are kept separate, performance stability is prioritized, and outputs are designed to be interpretable rather than opaque.
