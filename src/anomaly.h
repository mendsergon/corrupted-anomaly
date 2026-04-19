#ifndef ANOMALY_H
#define ANOMALY_H

#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

// ============================================================================
// Configuration (all overridable via command line)
// ============================================================================

struct OrbConfig {
    float shellRadius   = 1.0f;    // nominal shell radius (centerline)
    float shellThick    = 0.04f;   // shell thickness (thin, modulated thinner near holes)
    float dispAmp       = 0.12f;   // low-freq FBM mountain amplitude
    float spikeAmp      = 0.06f;   // high-freq mini-spike amplitude (gated)
    float holeRadius    = 0.30f;   // hole subtractor sphere radius (base)
    int   holeCount     = 24;      // number of drifting holes (sizes vary)
    float holeDrift     = 0.10f;   // base drift speed (per-hole hash scaling applied)
    float coreRadius    = 0.22f;   // central core radius
    int   veinCount     = 24;      // number of vein fibers
    float veinR0        = 0.018f;  // vein radius at core end
    float veinR1        = 0.006f;  // vein radius at membrane end
    float veinCurve     = 0.18f;   // perpendicular midpoint offset
    int   spikeCount    = 96;      // total spike count (solo + clusters + small)
    float spikeHeight   = 0.36f;   // max peak height of a spike
    float spikeWidth    = 0.14f;   // spike base half-angle (radians)
    float spikeSharp    = 5.5f;    // cone profile exponent
    float spikeDrift    = 0.12f;   // per-spike axis rotation rate
    float spikeRate     = 0.35f;   // per-spike height oscillation rate
    int   maxSteps      = 128;
    int   windowWidth   = 1280;
    int   windowHeight  = 720;
    float camDistance   = 3.2f;
    float camTheta      = 1.32f;
    float camPhi        = 0.4f;
    float camFov        = 45.0f;
};

OrbConfig parseArgs(int argc, char** argv);

// ============================================================================
// Camera
// ============================================================================

struct Camera {
    float distance;
    float fov;

    // Orientation held as an orthonormal basis in world space.
    // `fwd` points from the camera toward the origin; `right` and `up`
    // span the camera's image plane. Mouse drag rotates the whole basis
    // via axis-angle -- no Euler angles, no pole singularity.
    cl_float3 right;
    cl_float3 up;
    cl_float3 fwd;

    // smoothing targets (orientation updates immediately, not smoothed)
    float targetDistance;
    float targetFov;

    // drag state
    double lastMouseX;
    double lastMouseY;
    bool   dragging;
};

Camera createDefaultCamera(const OrbConfig& cfg);

// ============================================================================
// GPU Raymarcher
// ============================================================================
// One class, one OpenCL kernel, writes float RGBA into an internal buffer
// that main.cpp uploads to a GL texture each frame. Matches the Kerr
// architecture (GPURayTracer in blackhole.h/.cpp).

class GPURaymarcher {
public:
    GPURaymarcher();
    ~GPURaymarcher();

    bool initialize(const OrbConfig& cfg);
    void cleanup();

    // Run the kernel. After this call getPixels() has width*height float4s.
    void renderFrame(const Camera& cam, int width, int height,
                     float simTime, float amp);

    std::vector<float>& getPixels() { return m_pixels; }

    bool isAvailable() const { return m_initialized; }
    std::string getDeviceInfo() const { return m_deviceInfo; }
    double getLastRenderMs() const { return m_lastRenderMs; }

private:
    bool m_initialized = false;
    std::string m_deviceInfo;
    double m_lastRenderMs = 0.0;

    std::vector<float> m_pixels;

    cl_platform_id   m_platform = nullptr;
    cl_device_id     m_device   = nullptr;
    cl_context       m_context  = nullptr;
    cl_command_queue m_queue    = nullptr;
    cl_program       m_program  = nullptr;
    cl_kernel        m_kernel   = nullptr;

    cl_mem m_pixelBuffer = nullptr;
    int    m_bufW = 0, m_bufH = 0;

    bool   selectBestDevice();
    bool   createContext();
    bool   buildProgram(const OrbConfig& cfg);
    bool   createBuffers(int w, int h);

    std::string loadKernelSource(const std::string& filename);
};

// ============================================================================
// Global instances (defined in anomaly.cpp)
// ============================================================================

extern GPURaymarcher g_raymarcher;
extern OrbConfig     g_config;

// ============================================================================
// Post-processing (CPU-side, run on GPU's output float RGBA buffer)
// ============================================================================

void applyBloom(std::vector<float>& pixels, int width, int height);

// Save PPM screenshot to Screenshots/ directory
void saveScreenshot(const std::vector<float>& pixels, int width, int height);

#endif
