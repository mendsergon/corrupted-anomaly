# Corrupted Anomaly

![version](https://img.shields.io/badge/version-1.0.0-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![gpu](https://img.shields.io/badge/GPU-OpenCL%201.2-orange)

A single-kernel OpenCL compute raymarcher rendering a procedurally-generated hollow organic anomaly: a breathing core pumping cardiac waves through a web of 48 curved veins attached to a perforated membrane, with drifting holes, shifting spike clusters, and thin cross-connections. Built from scratch in C++/OpenCL with no rendering libraries — GLFW for the window, GL compat-profile for the final texture blit, everything else is math in a `.cl` file. Designed and tuned on AMD RDNA4 (RX 9070 XT), Mesa 26.x.

## Features

- Fully procedural hollow-membrane SDF scene: outer shell with displaced mountains, perforated by 24 shifting non-circular holes; inner core pulsing on a cardiac waveform; 48 curved veins connecting 6 core hubs to 8 membrane regions; 20 bridge cross-connections; 96 dynamically emerging/receding spikes.
- Cardiac simulation: two-gaussian S1/S2 waveform drives core radius; phase-locked traveling radius pulse on veins gives visible blood-flow effect emerging from the core at each systole.
- Continuous-gradient hole avoidance (no teleports): smoothstep-based push fields guarantee C¹ continuity of vein endpoint positions as holes drift.
- Closed-form spherical arc-visibility test for vein routing: the region-to-endpoint great-circle arc is kept clear of hole cones via a gradient-flow adjustment, no graph search needed.
- Free-tumble orbit camera: quaternion-style orthonormal basis with Rodrigues rotation on mouse drag — rotate continuously through any pole with zero singularities.
- Zoom range allows camera to sit inside the hollow looking at the core and vein web from the inside.
- Runtime toggles: diagnostic shading modes (grey Lambert / normals / structure tags), bloom, pause, reset, screenshot.

## Getting started

### Dependencies

**Arch Linux:**
```bash
sudo pacman -S gcc make glfw opencl-headers ocl-icd
# AMD ROCm / Mesa OpenCL driver of your choice
```

**Ubuntu / Debian:**
```bash
sudo apt install build-essential libglfw3-dev opencl-headers ocl-icd-opencl-dev
```

**Fedora / RHEL:**
```bash
sudo dnf install gcc-c++ make glfw-devel opencl-headers ocl-icd-devel
```

### Build and run

```bash
git clone git@github.com:mendsergon/corrupted-anomaly.git
cd corrupted-anomaly
make
./anomaly
```

Run from the project root so the kernel source at `src/anomaly.cl` resolves at runtime.

## Controls

| Input | Action |
|---|---|
| `LMB` + drag | Free-tumble orbit camera (continuous through poles) |
| Scroll wheel | Zoom (min 0.28: inside the hollow; max 25.0) |
| `R` | Reset camera to default view |
| `Space` | Pause animation |
| `B` | Toggle bloom |
| `F12` | Save screenshot (PPM) |
| `+` / `-` | Adjust FoV |
| `ESC` / `Q` | Quit |

## Command-line flags

All scene parameters are exposed as CLI overrides. The important ones:

```bash
./anomaly --width 1280 --height 720       # resolution
./anomaly --veins 48 --holes 24           # primitive counts
./anomaly --thick 0.04 --disp 0.12        # membrane thickness / displacement
./anomaly --spike-h 0.36 --spikes 96      # spike parameters
./anomaly --distance 3.2 --fov 60         # initial camera
```

See `anomaly.cpp::parseArgs` for the complete list.

## How it works

### Architecture

One OpenCL kernel produces a float4 RGBA buffer on the GPU per frame. The host copies the buffer to a GL texture via `glTexSubImage2D` and blits it to the window with an immediate-mode fullscreen quad under GL 3.3 compatibility profile. This mirrors the architecture of the accompanying Kerr black hole raytracer — minimal interop, maximum simplicity.

At entry the kernel builds a per-pixel `SceneCtx` (holes, core hubs, veins with 3 interior control points each, bridges, spikes) from the current sim time `t`, then sphere-traces the signed distance function for the pixel's primary ray. Hit points are shaded with a half-Lambert grey against a fixed directional light; miss rays return black.

### Scene composition

The SDF is built in layers:

- **Outer shell.** Two offset spheres at `R_SHELL ± THICK/2 + displacement(dir, t)`, combined as `max(d_outer, −d_inner)` for a hollow membrane.
- **Mountain displacement.** 3-octave FBM for low-frequency surface relief; separate gated high-frequency layer for surface roughness; 96 spike cones for dramatic protrusions.
- **Hole carving.** 24 angular cones subtracted from the shell, each with its own tangent-plane basis and two-sinusoid boundary perturbation for non-circular edges. Regional coverage capped at 1/3 per octant via softmax Voronoi scaling.
- **Core.** Sphere at `R_CORE` with radius modulated by the cardiac pump waveform.
- **Veins.** 48 4-segment curved capsules through 3 interior control points, radius tapered and modulated by a static noise + a traveling cardiac pulse. Smin-joined at segment junctions and smin-blended with hubs.
- **Hubs.** 6 small spheres on the core surface, each shared as the start anchor for 8 veins. Smin-melted into vein SDF.
- **Bridges.** 20 thin constant-radius capsules between vein midpoints.

Final combine: `smin_poly(d_shell_carved, d_inside, FUSE_K)` — the vein tips fuse smoothly into the inner membrane rather than bolting onto it.

### Key technical choices

Four non-obvious engineering decisions worth calling out:

**1. Continuous-gradient hole avoidance on the sphere.** For a unit direction `d` and a hole with center `h` and angular radius `hole_ang`, the push is

```
overlap = smoothstep(cos(hole_ang) − 0.12, cos(hole_ang) + 0.02, dot(d, h))
push   += − normalize(d − h·dot(d,h)) · overlap · 0.20
```

summed over all holes, iterated 3 passes, renormalized. Every term is C¹ in the hole positions, so as a hole drifts across a vein endpoint the endpoint slides continuously around it — no discrete switching, no teleport. This replaces the classic "pick the worst violator and snap to the boundary" pattern, which is discontinuous at the moment the worst violator identity changes.

**2. Closed-form great-circle arc visibility.** The great-circle arc from region center `a` to endpoint `b` is parameterized as `arc(α) = a·cos(α) + t·sin(α)` for `α ∈ [0, θ]`, where `t` is the unit tangent at `a` aimed at `b` and `θ = acos(dot(a,b))`. The penetration function against a hole is `f(α) = c_r·cos(α) + c_t·sin(α)` with `c_r = dot(a, hole_dir)`, `c_t = dot(t, hole_dir)`. Its critical point is `α* = atan2(c_t, c_r)` with `f(α*) = √(c_r² + c_t²)`. So the maximum arc penetration per hole is ~15 float ops — no graph search, no sampling, no CGAL. Used both as a smooth arc-escape gradient and (in the visibility predicate) to constrain vein endpoint selection.

**3. Regional allocation analogous to the hole softmax Voronoi.** Vein endpoints are hard-assigned to 8 fibonacci-distributed region centers on the membrane (`region = i mod 8`, `local = i div 8`). Within each region, the 6 members spread in a flower pattern via distinct tangent-plane offsets at varying angular magnitudes (40% to 100% of `VEIN_SPREAD`). Clean arithmetic (48 = 8·6), natural clustering without needing softmax — the softmax was necessary for holes because hole assignments were unequal in count, but vein counts per region are equal by construction.

**4. Phase-locked cardiac and flow model.** Core radius carries a two-gaussian S1/S2 waveform at `HEART_RATE` (0.60 Hz by default, the slow-dramatic range). Vein radius carries a traveling pulse `(cos(2π·(h_global − t·HEART_RATE)) + 1)/2` where `h_global = seg_start + h_local · seg_extent` is a parameter continuous across the vein's 4 segments. Since the two waveforms share `HEART_RATE`, every core contraction coincides with a pulse emerging from the core hubs and traveling outward along each vein — visually lock-step.

### Camera

A quaternion-style orthonormal basis `(right, up, fwd)` held in world space, updated on mouse drag via Rodrigues rotation around the current `up` (horizontal drag) and current `right` (vertical drag) axes. Re-orthonormalised every frame. No Euler angles, no world-up switching, no pole singularity — can rotate freely through any orientation.

## Project structure

```
corrupted-anomaly/
├── Makefile              distro autodetect + check-assets + deps
├── README.md
├── .gitignore
└── src/
    ├── anomaly.h         OrbConfig, Camera, GPURaymarcher class
    ├── anomaly.cpp       OpenCL setup, CLI parser, camera factory
    ├── main.cpp          GLFW + GL fullscreen quad + input loop
    └── anomaly.cl        single-kernel SDF + shading
```

## Limitations

- Single-GPU, single-platform (Linux + AMD-centric, tested on RDNA4). Probably works on NVIDIA and Intel OpenCL implementations but not validated.
- Sphere-trace safety factor is 0.50 — conservative, leaves performance on the table. Some radius-modulated primitives are Lipschitz bounds rather than true distances.
- No material pass. Surface shading is half-Lambert grey against a fixed light; there is no colour, no subsurface scattering, no PBR. Shape-first by design; materials are future work.
- `build_ctx` currently runs per-pixel despite producing identical output for all pixels in a frame. Correctly-written waste but waste.

## Future work

- Move `build_ctx` to host-side C++, upload once per frame as a `__constant` buffer. Large win for resolution scaling.
- Per-point bounding-sphere culling of veins / bridges / hubs. Most march-step SDF evaluations only interact with a handful of primitives; the rest are redundant.
- Per-point hole and spike culling: skip the tangent-frame noise computation for holes angularly far from the ray point, same for spike cones.
- Materials pass: basic colour gradients, subsurface for the membrane, emissive core. Currently deferred because shape was the hard part.
- CL-GL interop: `clCreateFromGLTexture` to skip the host-bounce copy.

## License

MIT. Do anything you want with the code.
