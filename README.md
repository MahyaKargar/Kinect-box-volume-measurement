# Kinect-Based Box Dimension & Volume Estimation

Estimating the dimensions and volume of rectangular box-shaped objects from depth data captured by a **Microsoft Kinect v1**, and characterizing how ambient light affects sensor performance — all in Python.

---

## Overview

This project uses the Kinect v1's structured-light IR depth sensor to automatically measure the **width, length, height, and volume** of a rectangular box placed in front of the camera, without any manual measurement. It also includes a dedicated experiment measuring how the sensor's accuracy changes under different ambient lighting conditions.

The pipeline works in three broad stages:

1. **Depth acquisition & differencing** — capture multiple reference (empty-scene) depth frames, average them for stability, and subtract the resulting reference from a frame containing the object to isolate the object's silhouette.
2. **Adaptive filtering & 3D reconstruction** — clean the depth difference mask, remove outliers and flying pixels, and reconstruct a 3D point cloud of the object's top surface.
3. **Dimension & volume estimation** — fit an oriented bounding box to the surface, correct for camera tilt, and compute the object's dimensions and volume via a footprint (area × height) method.

---

## Key Features

- Automatic width / length / height estimation of rectangular boxes from reference and object depth frames (the reference frame is averaged over multiple captures for stability)
- Volume estimation via a footprint (cross-sectional area × height) method
- Oriented bounding box fitting (`minAreaRect`) — works even when the box isn't axis-aligned with the camera
- Robust outlier rejection using a **MAD (Median Absolute Deviation)** filter, with a configurable sensitivity factor
- Empirically calibrated **cosine-based tilt correction** to remove systematic height bias caused by the camera's mounting angle
- A brightness-based ambient-light experiment using the Kinect's RGB stream as a proxy for lighting conditions, to find the sensor's optimal operating light range
- CSV logging of all experimental results for later analysis

---

## Architecture

The codebase is organized into focused, single-responsibility modules:

| Module Responsibility |                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------- |
| `KinectCamera`        | Low-level interface to the Kinect v1 (depth + RGB streams)                            |
| `DepthProcessor`      | Depth-frame differencing, adaptive thresholding, MAD-based outlier filtering          |
| `ObjectAnalyzer`      | Top-surface extraction, oriented bounding box fitting, connected-component selection  |
| `PointCloudBuilder`   | Converts the filtered depth mask into a 3D point cloud using the pinhole camera model |
| `VolumeCalculator`    | Applies tilt correction and computes final dimensions and volume                      |
| `CSVLogger`           | Logs measurement and lighting-test results to CSV                                     |
| `AppController`       | Orchestrates the full pipeline end-to-end                                             |
| `main.py`             | Entry point                                                                           |

---

## Methodology Highlights

- **Depth-frame differencing** isolates the object by comparing an averaged reference (empty-scene) depth frame against the frame containing the object.
- **Adaptive thresholding** models the sensor's expected depth resolution as a function of distance (disparity quantization), so filtering thresholds scale correctly at different ranges.
- **MAD outlier filtering** removes "flying pixels" and other depth noise around object edges. Default sensitivity factor: `mad_factor = 2` (increased to `3` for smaller objects, which are more sensitive to noise).
- **Tilt correction** — raw depth differences measure distance along the camera's optical axis, not true vertical height. A cosine-based correction factor, empirically calibrated to the camera's mounting angle, removes this systematic bias.
- **Volume estimation** uses a footprint method: cross-sectional area (derived from the pinhole camera model) multiplied by the tilt-corrected representative height of the object.
- **Ambient light proxy** — instead of a physical lux meter, the average luma brightness of the Kinect's own RGB stream is used to characterize lighting conditions during the light-sensitivity experiment.

---

## Requirements

- Python 3.10
- [OpenCV](https://opencv.org/) — `minAreaRect`, connected-component labeling, morphological operations
- [NumPy](https://numpy.org/)
- Kinect v1 driver/SDK for depth + RGB streams (e.g. `freenect` / `libfreenect`, or **Microsoft Kinect for Windows SDK 1.8** on Windows)

---

## Limitations & Future Work

- Currently limited to single, isolated rectangular box-shaped objects
- Performance depends on ambient lighting conditions
- Future directions: multi-object support, automatic camera-angle calibration, and testing on newer depth sensors (e.g. Kinect v2, Azure Kinect)

---

## Author

Mahya Kargar — University capstone project
