FlowSync - Computer Vision Module Handoff

Purpose

This document defines the Computer Vision (CV) module of FlowSync and the interface between the CV pipeline and the downstream gesture-recognition/AI module.

The CV module converts webcam frames into a consistent numerical representation of the user's hand.

CV Pipeline

Webcam Frame
     |
     v
Camera Input / OpenCV
     |
     v
MediaPipe Hand Detection & Tracking
     |
     v
21 Hand Landmarks
     |
     v
Landmark Validation
     |
     v
Landmark Normalization
     |
     v
Geometric Feature Extraction
     |
     v
Consistent Numerical Representation
     |
     v
Gesture Recognition / AI Module

Module Responsibilities

1. Camera Input

The camera layer is responsible for capturing real-time frames from the webcam and providing frames to the hand-tracking pipeline.

2. Hand Detection and Tracking

MediaPipe Hands is used to detect and track the user's hand from incoming frames.

For a valid detected hand, the module extracts 21 hand landmarks. Each landmark provides positional information in the form of x, y, and z coordinates.

3. Landmark Processing

The extracted landmarks are converted into a consistent internal representation. Landmark ordering must remain fixed so that downstream processing receives the same feature meaning for every frame.

4. Handling Missing or Incomplete Landmarks

The CV pipeline must safely handle frames where:

No hand is detected.

Tracking is temporarily lost.

Landmark information is incomplete or invalid.

A temporary failure should not crash the application. The pipeline should return the agreed invalid/empty state and continue processing subsequent frames.

5. Normalization

Landmark coordinates are normalized to reduce the effect of differences in hand position, scale, and camera distance.

The goal is to make the representation more consistent across users and frames.

6. Geometric Feature Extraction

Where required, geometric information is derived from the landmarks, such as:

Relative positions

Landmark distances

Angles

Direction/vector information

Only the features required by the agreed recognition pipeline should be exposed to the downstream module.

Repository Structure

Current CV repository structure:

FlowSync/
│
├── app/
├── Camera/
├── features/
├── hand_tracking/
├── Landmarks/
├── venv311/
│
├── .gitignore
├── CV_HANDOFF.md
├── hand_landmarker.task
├── main.py
├── README.md
└── requirements.txt

The exact implementation inside each folder may evolve, but the functional responsibilities should remain separated.

Key Technologies

Python - Main implementation language

OpenCV - Webcam access and frame handling

MediaPipe Hands - Hand detection, tracking, and 21-landmark extraction

NumPy - Numerical processing and feature calculations

CV Output Contract

The CV module is a data provider for the downstream gesture-recognition module.

Input

Webcam video frame

Processing

Frame
 -> Hand detection
 -> 21 landmarks
 -> Validation
 -> Normalization
 -> Feature extraction

Output

A consistent numerical representation of the detected hand.

The downstream module should not need to know how the camera or MediaPipe pipeline is implemented internally.

Integration Requirements

The following conventions must remain fixed between the CV and AI/gesture-recognition modules:

Landmark ordering

Coordinate convention

Normalization procedure

Feature ordering

Output dimensions

Missing-hand / invalid-frame behavior

Any change to the output representation should be communicated before the downstream model or recognition logic is updated.

Example Integration Flow

frame
  |
  v
CV Module
  |
  v
processed_features
  |
  v
Gesture Recognition
  |
  v
gesture_label

The recognition module consumes the processed representation and should not directly depend on raw webcam frames unless explicitly agreed by the team.

Calibration Support

The CV module supports user-specific calibration where required.

Calibration is intended to capture user-related characteristics such as hand position, scale, movement range, or relevant gesture thresholds.

User
  |
  v
Calibration
  |
  v
User-specific parameters
  |
  v
CV processing / gesture interpretation

Calibration is separate from the core hand-detection process and should not change the fixed landmark ordering.

Testing Status

The CV module has already been tested during development for the core implementation, including:

Camera/frame acquisition

Hand detection and tracking

21-landmark extraction

Landmark processing

Normalization

Missing-hand handling

Feature generation

Calibration-related processing

The next major validation point is integration with the downstream gesture-recognition module.

Handoff Boundary

CV Module owns

Webcam input

Frame handling

Hand detection/tracking

21 landmarks

Landmark validation

Normalization

Geometric feature extraction

Standardized numerical output

Calibration support

Downstream gesture-recognition module owns

Gesture classification/recognition logic

Model training/inference

Gesture label generation

Computer-interaction module owns

Mapping recognized gestures to computer actions

Cursor movement

Clicking

Scrolling

Dragging

Other OS-level actions

Integration Goal

The CV module is considered integration-ready when the downstream module can receive the standardized output without needing to know the internal implementation details of camera capture or MediaPipe.

CV Module
   |
   | standardized numerical representation
   v
Gesture Recognition
   |
   | recognized gesture
   v
Action Mapping
   |
   v
Computer Interaction

Maintainer

Member 2 - Computer Vision Engineer

Primary responsibilities:

Camera acquisition

MediaPipe hand tracking

21-landmark extraction

Landmark processing

Normalization

Geometric feature extraction

Missing/incomplete landmark handling

Calibration support

CV-to-AI integration