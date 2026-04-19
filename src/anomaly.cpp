#include "anomaly.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

GPURaymarcher g_raymarcher;
OrbConfig     g_config;

// ============================================================================
// GPURaymarcher
// ============================================================================

GPURaymarcher::GPURaymarcher() {}
GPURaymarcher::~GPURaymarcher() { cleanup(); }

bool GPURaymarcher::initialize(const OrbConfig& cfg) {
    std::cout << "Initializing OpenCL GPU raymarcher..." << std::endl;
    if (!selectBestDevice()) {
        std::cerr << "No suitable OpenCL GPU." << std::endl;
        return false;
    }
    if (!createContext()) return false;
    if (!buildProgram(cfg)) { cleanup(); return false; }
    m_initialized = true;
    std::cout << "GPU raymarcher enabled: " << m_deviceInfo << std::endl;
    return true;
}

void GPURaymarcher::cleanup() {
    if (m_kernel)      clReleaseKernel(m_kernel);
    if (m_program)     clReleaseProgram(m_program);
    if (m_pixelBuffer) clReleaseMemObject(m_pixelBuffer);
    if (m_queue)       clReleaseCommandQueue(m_queue);
    if (m_context)     clReleaseContext(m_context);
    m_kernel = nullptr;
    m_program = nullptr;
    m_pixelBuffer = nullptr;
    m_queue = nullptr;
    m_context = nullptr;
    m_initialized = false;
}

bool GPURaymarcher::selectBestDevice() {
    cl_uint nPlat = 0;
    if (clGetPlatformIDs(0, nullptr, &nPlat) != CL_SUCCESS || nPlat == 0) return false;
    std::vector<cl_platform_id> plats(nPlat);
    clGetPlatformIDs(nPlat, plats.data(), nullptr);

    cl_device_id   bestDev  = nullptr;
    cl_platform_id bestPlat = nullptr;
    int            bestScore = -1;

    for (auto p : plats) {
        cl_uint nDev = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nDev) != CL_SUCCESS) continue;
        if (nDev == 0) continue;
        std::vector<cl_device_id> devs(nDev);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nDev, devs.data(), nullptr);

        for (auto d : devs) {
            char name[256] = {0};
            char vendor[256] = {0};
            cl_uint cu = 0;
            clGetDeviceInfo(d, CL_DEVICE_NAME,   sizeof(name),   name,   nullptr);
            clGetDeviceInfo(d, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
            clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);

            int score = (int)cu;
            std::string v(vendor);
            if (v.find("AMD") != std::string::npos || v.find("Advanced Micro") != std::string::npos)
                score += 1000;
            else if (v.find("NVIDIA") != std::string::npos) score += 1000;
            else if (v.find("Intel")  != std::string::npos) score += 100;

            std::cout << "  Found: " << name << " (" << vendor << ") CUs=" << cu
                      << " score=" << score << std::endl;

            if (score > bestScore) {
                bestScore = score;
                bestDev   = d;
                bestPlat  = p;
                m_deviceInfo = std::string(name) + " (" + vendor + ")";
            }
        }
    }
    if (!bestDev) return false;
    m_platform = bestPlat;
    m_device   = bestDev;
    return true;
}

bool GPURaymarcher::createContext() {
    cl_int err;
    m_context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "createContext failed: " << err << std::endl; return false; }
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    m_queue = clCreateCommandQueueWithProperties(m_context, m_device, props, &err);
#else
    m_queue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    if (err != CL_SUCCESS) { std::cerr << "createCommandQueue failed: " << err << std::endl; return false; }
    return true;
}

std::string GPURaymarcher::loadKernelSource(const std::string& filename) {
    std::vector<std::string> paths = {
        filename, "./" + filename, "src/" + filename, "../" + filename, "../src/" + filename
    };
    for (const auto& p : paths) {
        std::ifstream f(p);
        if (f.is_open()) { std::stringstream ss; ss << f.rdbuf(); return ss.str(); }
    }
    std::cerr << "Could not locate " << filename << std::endl;
    return "";
}

bool GPURaymarcher::buildProgram(const OrbConfig& cfg) {
    std::string src = loadKernelSource("anomaly.cl");
    if (src.empty()) return false;

    cl_int err;
    const char* srcPtr = src.c_str();
    size_t srcLen = src.length();
    m_program = clCreateProgramWithSource(m_context, 1, &srcPtr, &srcLen, &err);
    if (err != CL_SUCCESS) return false;

    char opts[1024];
    std::snprintf(opts, sizeof(opts),
        "-cl-fast-relaxed-math -cl-mad-enable "
        "-DR_SHELL=%.4ff -DTHICK=%.4ff -DDISP_AMP=%.4ff -DSPIKE_AMP=%.4ff "
        "-DR_HOLE=%.4ff -DN_HOLES=%d -DHOLE_DRIFT=%.4ff "
        "-DR_CORE=%.4ff -DN_VEINS=%d "
        "-DVEIN_R0=%.4ff -DVEIN_R1=%.4ff -DVEIN_CURVE=%.4ff "
        "-DN_SPIKES=%d -DSPIKE_H=%.4ff -DSPIKE_W=%.4ff "
        "-DSPIKE_SHARP=%.4ff -DSPIKE_DRIFT=%.4ff -DSPIKE_RATE=%.4ff "
        "-DMAX_STEPS=%d",
        cfg.shellRadius, cfg.shellThick, cfg.dispAmp, cfg.spikeAmp,
        cfg.holeRadius, cfg.holeCount, cfg.holeDrift,
        cfg.coreRadius, cfg.veinCount,
        cfg.veinR0, cfg.veinR1, cfg.veinCurve,
        cfg.spikeCount, cfg.spikeHeight, cfg.spikeWidth,
        cfg.spikeSharp, cfg.spikeDrift, cfg.spikeRate,
        cfg.maxSteps);

    err = clBuildProgram(m_program, 1, &m_device, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSz = 0;
        clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSz);
        std::vector<char> log(logSz + 1);
        clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, logSz, log.data(), nullptr);
        std::cerr << "OpenCL build failed:\n" << log.data() << std::endl;
        return false;
    }

    m_kernel = clCreateKernel(m_program, "raymarch_anomaly", &err);
    if (err != CL_SUCCESS) { std::cerr << "clCreateKernel failed: " << err << std::endl; return false; }
    return true;
}

bool GPURaymarcher::createBuffers(int w, int h) {
    if (w == m_bufW && h == m_bufH && m_pixelBuffer) return true;
    if (m_pixelBuffer) { clReleaseMemObject(m_pixelBuffer); m_pixelBuffer = nullptr; }
    cl_int err;
    size_t sz = (size_t)w * (size_t)h * sizeof(cl_float4);
    m_pixelBuffer = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, sz, nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "pixelBuffer create failed: " << err << std::endl; return false; }
    m_bufW = w; m_bufH = h;
    return true;
}

void GPURaymarcher::renderFrame(const Camera& cam, int width, int height,
                                 float simTime, float amp) {
    if (!m_initialized) return;
    if (!createBuffers(width, height)) return;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Camera orientation is already an orthonormal basis in world space --
    // no spherical-coordinate math, no pole-safe reference switching. eye
    // is just -fwd * distance away from the origin.
    cl_float3 eye   = { -cam.fwd.s[0] * cam.distance,
                        -cam.fwd.s[1] * cam.distance,
                        -cam.fwd.s[2] * cam.distance };
    cl_float3 fwd   = cam.fwd;
    cl_float3 right = cam.right;
    cl_float3 up    = cam.up;

    float fov_rad = cam.fov * 3.14159265f / 180.0f;
    float fov_tan = std::tan(fov_rad * 0.5f);

    int num_sites = 0;  // unused in new kernel; kept to preserve layout
    (void)num_sites;

    // Diagnostic mode (set once from ANOMALY_DIAG env var). 0 = normal grey
    // Lambert; 1 = normals as RGB; 2 = hex distance field; 3 = structure tag.
    static int diag_mode = -1;
    if (diag_mode < 0) {
        const char* s = std::getenv("ANOMALY_DIAG");
        diag_mode = s ? std::atoi(s) : 0;
        if (diag_mode != 0)
            std::printf("[diag] ANOMALY_DIAG=%d (1=normals 2=hex 3=structure)\n", diag_mode);
    }

    cl_int err = 0;
    err |= clSetKernelArg(m_kernel,  0, sizeof(cl_mem),    &m_pixelBuffer);
    err |= clSetKernelArg(m_kernel,  1, sizeof(int),       &width);
    err |= clSetKernelArg(m_kernel,  2, sizeof(int),       &height);
    err |= clSetKernelArg(m_kernel,  3, sizeof(cl_float3), &eye);
    err |= clSetKernelArg(m_kernel,  4, sizeof(cl_float3), &fwd);
    err |= clSetKernelArg(m_kernel,  5, sizeof(cl_float3), &right);
    err |= clSetKernelArg(m_kernel,  6, sizeof(cl_float3), &up);
    err |= clSetKernelArg(m_kernel,  7, sizeof(float),     &fov_tan);
    err |= clSetKernelArg(m_kernel,  8, sizeof(float),     &simTime);
    err |= clSetKernelArg(m_kernel,  9, sizeof(float),     &amp);
    err |= clSetKernelArg(m_kernel, 10, sizeof(int),       &diag_mode);
    if (err != CL_SUCCESS) { std::cerr << "clSetKernelArg failed: " << err << std::endl; return; }

    size_t total = (size_t)width * (size_t)height;
    size_t local = 256;
    size_t global = ((total + local - 1) / local) * local;

    err = clEnqueueNDRangeKernel(m_queue, m_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { std::cerr << "kernel launch failed: " << err << std::endl; return; }
    clFinish(m_queue);

    m_pixels.resize((size_t)width * (size_t)height * 4);
    err = clEnqueueReadBuffer(m_queue, m_pixelBuffer, CL_TRUE, 0,
                               total * sizeof(cl_float4), m_pixels.data(),
                               0, nullptr, nullptr);
    if (err != CL_SUCCESS) std::cerr << "readback failed: " << err << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    m_lastRenderMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ============================================================================
// Camera factory
// ============================================================================

Camera createDefaultCamera(const OrbConfig& cfg) {
    Camera c;
    c.distance = cfg.camDistance;
    c.fov      = cfg.camFov;
    c.targetDistance = c.distance;
    c.targetFov     = c.fov;
    c.lastMouseX = 0.0;
    c.lastMouseY = 0.0;
    c.dragging   = false;

    // Convert the one-time spherical (theta, phi, distance) from the config
    // into the orthonormal basis we actually keep. From here on, only the
    // basis is authoritative; theta/phi are gone.
    float st = std::sin(cfg.camTheta), ct = std::cos(cfg.camTheta);
    float sp = std::sin(cfg.camPhi),   cp = std::cos(cfg.camPhi);
    cl_float3 eye = { cfg.camDistance * st * sp,
                      cfg.camDistance * ct,
                      cfg.camDistance * st * cp };
    float elen = std::sqrt(eye.s[0]*eye.s[0] + eye.s[1]*eye.s[1] + eye.s[2]*eye.s[2]);
    c.fwd = { -eye.s[0] / elen, -eye.s[1] / elen, -eye.s[2] / elen };

    // Initial up reference: world Y, unless the starting view points nearly
    // straight up or down, in which case pick Z to avoid a degenerate cross.
    cl_float3 up_ref = { 0.0f, 1.0f, 0.0f };
    if (std::fabs(c.fwd.s[1]) > 0.99f) up_ref = { 0.0f, 0.0f, 1.0f };

    // right = normalize(up_ref × fwd)
    cl_float3 rraw = {
        up_ref.s[1] * c.fwd.s[2] - up_ref.s[2] * c.fwd.s[1],
        up_ref.s[2] * c.fwd.s[0] - up_ref.s[0] * c.fwd.s[2],
        up_ref.s[0] * c.fwd.s[1] - up_ref.s[1] * c.fwd.s[0]
    };
    float rlen = std::sqrt(rraw.s[0]*rraw.s[0] + rraw.s[1]*rraw.s[1] + rraw.s[2]*rraw.s[2]);
    c.right = { rraw.s[0]/rlen, rraw.s[1]/rlen, rraw.s[2]/rlen };

    // up = fwd × right (gives a positively-oriented, already-unit basis)
    c.up = {
        c.fwd.s[1] * c.right.s[2] - c.fwd.s[2] * c.right.s[1],
        c.fwd.s[2] * c.right.s[0] - c.fwd.s[0] * c.right.s[2],
        c.fwd.s[0] * c.right.s[1] - c.fwd.s[1] * c.right.s[0]
    };

    return c;
}

// ============================================================================
// CLI
// ============================================================================

OrbConfig parseArgs(int argc, char** argv) {
    OrbConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::cout
                << "Corrupted Arcane Anomaly (shape-first SDF raymarcher)\n"
                << "  --radius <f>      shell nominal radius    (default 1.0)\n"
                << "  --thick <f>       shell thickness         (default 0.04)\n"
                << "  --disp <f>        mountain amplitude      (default 0.12)\n"
                << "  --spike <f>       mini-spike amplitude    (default 0.06)\n"
                << "  --hole-r <f>      hole radius             (default 0.30)\n"
                << "  --holes <n>       number of holes         (default 24)\n"
                << "  --hole-drift <f>  hole drift speed rad/s  (default 0.10)\n"
                << "  --core <f>        core radius             (default 0.22)\n"
                << "  --veins <n>       number of veins         (default 24)\n"
                << "  --vein-r0 <f>     vein radius at core end (default 0.018)\n"
                << "  --vein-r1 <f>     vein radius at tip      (default 0.006)\n"
                << "  --vein-curve <f>  vein midpoint offset    (default 0.18)\n"
                << "  --spikes <n>      spike count             (default 96, split small+solo+clusters)\n"
                << "  --spike-h <f>     max spike height        (default 0.36)\n"
                << "  --spike-w <f>     spike base half-angle   (default 0.14)\n"
                << "  --spike-sharp <f> cone exponent           (default 5.5)\n"
                << "  --spike-drift <f> spike drift rate        (default 0.12)\n"
                << "  --spike-rate <f>  spike height osc rate   (default 0.35)\n"
                << "  --steps <n>       max sphere-trace iters  (default 128)\n"
                << "  --width <n>       window width            (default 1280)\n"
                << "  --height <n>      window height           (default 720)\n"
                << "  --distance <f>    initial camera distance (default 3.2)\n"
                << "  --theta <deg>     initial camera polar    (default 75.6)\n"
                << "  --fov <deg>       initial field of view   (default 45)\n"
                << "\n"
                << "env var ANOMALY_DIAG:\n"
                << "  0 (default) : grey matte half-Lambert\n"
                << "  1           : normals as RGB\n"
                << "  2           : structure tag (red shell, green core, blue veins)\n";
            std::exit(0);
        }
        if (i + 1 >= argc) continue;
        if      (a == "--radius")     cfg.shellRadius  = std::stof(argv[++i]);
        else if (a == "--thick")      cfg.shellThick   = std::stof(argv[++i]);
        else if (a == "--disp")       cfg.dispAmp      = std::stof(argv[++i]);
        else if (a == "--spike")      cfg.spikeAmp     = std::stof(argv[++i]);
        else if (a == "--hole-r")     cfg.holeRadius   = std::stof(argv[++i]);
        else if (a == "--holes")      cfg.holeCount    = std::stoi(argv[++i]);
        else if (a == "--hole-drift") cfg.holeDrift    = std::stof(argv[++i]);
        else if (a == "--core")       cfg.coreRadius   = std::stof(argv[++i]);
        else if (a == "--veins")      cfg.veinCount    = std::stoi(argv[++i]);
        else if (a == "--vein-r0")    cfg.veinR0       = std::stof(argv[++i]);
        else if (a == "--vein-r1")    cfg.veinR1       = std::stof(argv[++i]);
        else if (a == "--vein-curve") cfg.veinCurve    = std::stof(argv[++i]);
        else if (a == "--spikes")     cfg.spikeCount   = std::stoi(argv[++i]);
        else if (a == "--spike-h")    cfg.spikeHeight  = std::stof(argv[++i]);
        else if (a == "--spike-w")    cfg.spikeWidth   = std::stof(argv[++i]);
        else if (a == "--spike-sharp")cfg.spikeSharp   = std::stof(argv[++i]);
        else if (a == "--spike-drift")cfg.spikeDrift   = std::stof(argv[++i]);
        else if (a == "--spike-rate") cfg.spikeRate    = std::stof(argv[++i]);
        else if (a == "--steps")      cfg.maxSteps     = std::stoi(argv[++i]);
        else if (a == "--width")      cfg.windowWidth  = std::stoi(argv[++i]);
        else if (a == "--height")     cfg.windowHeight = std::stoi(argv[++i]);
        else if (a == "--distance")   cfg.camDistance  = std::stof(argv[++i]);
        else if (a == "--theta")      cfg.camTheta     = std::stof(argv[++i]) * 3.14159265f / 180.0f;
        else if (a == "--fov")        cfg.camFov       = std::stof(argv[++i]);
    }
    return cfg;
}

// ============================================================================
// CPU bloom (separable Gaussian, matches blackhole.cpp applyBloom)
// ============================================================================

void applyBloom(std::vector<float>& pixels, int width, int height) {
    int size = width * height * 4;
    std::vector<float> bright(size, 0.0f);
    const float thr = 0.70f;
    for (int i = 0; i < size; i += 4) {
        float lum = 0.2126f * pixels[i] + 0.7152f * pixels[i+1] + 0.0722f * pixels[i+2];
        if (lum > thr) {
            float k = (lum - thr) / (1.0f - thr);
            bright[i+0] = pixels[i+0] * k;
            bright[i+1] = pixels[i+1] * k;
            bright[i+2] = pixels[i+2] * k;
        }
    }

    int radius = std::max(2, width / 100);
    float sigma = (float)radius / 2.0f;
    std::vector<float> w(radius + 1);
    float ws = 0.0f;
    for (int i = 0; i <= radius; i++) {
        w[i] = std::exp(-(float)(i*i) / (2.0f * sigma * sigma));
        ws += (i == 0) ? w[i] : 2.0f * w[i];
    }
    for (int i = 0; i <= radius; i++) w[i] /= ws;

    std::vector<float> tmp(size, 0.0f);
    for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
        float r = 0, g = 0, b = 0;
        for (int k = -radius; k <= radius; k++) {
            int sx = std::max(0, std::min(width - 1, x + k));
            int i  = (y * width + sx) * 4;
            float wk = w[std::abs(k)];
            r += bright[i+0] * wk;
            g += bright[i+1] * wk;
            b += bright[i+2] * wk;
        }
        int i = (y * width + x) * 4;
        tmp[i+0] = r; tmp[i+1] = g; tmp[i+2] = b;
    }
    for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
        float r = 0, g = 0, b = 0;
        for (int k = -radius; k <= radius; k++) {
            int sy = std::max(0, std::min(height - 1, y + k));
            int i  = (sy * width + x) * 4;
            float wk = w[std::abs(k)];
            r += tmp[i+0] * wk;
            g += tmp[i+1] * wk;
            b += tmp[i+2] * wk;
        }
        int i = (y * width + x) * 4;
        bright[i+0] = r; bright[i+1] = g; bright[i+2] = b;
    }

    const float strength = 0.55f;
    for (int i = 0; i < size; i += 4) {
        pixels[i+0] = std::min(1.0f, pixels[i+0] + bright[i+0] * strength);
        pixels[i+1] = std::min(1.0f, pixels[i+1] + bright[i+1] * strength);
        pixels[i+2] = std::min(1.0f, pixels[i+2] + bright[i+2] * strength);
    }
}

// ============================================================================
// Screenshot (PPM, matches blackhole.cpp saveScreenshot)
// ============================================================================

void saveScreenshot(const std::vector<float>& pixels, int width, int height) {
    const std::string dir = "Screenshots";
#ifdef _WIN32
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0755);
#endif

    std::time_t now = std::time(nullptr);
    std::tm* t = std::localtime(&now);
    char filename[256];
    std::snprintf(filename, sizeof(filename),
                  "Screenshots/anomaly_%04d%02d%02d_%02d%02d%02d.ppm",
                  t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                  t->tm_hour, t->tm_min, t->tm_sec);

    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) { std::cerr << "screenshot open failed: " << filename << std::endl; return; }
    f << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; i++) {
        unsigned char r = (unsigned char)(std::clamp(pixels[i*4+0], 0.0f, 1.0f) * 255.0f);
        unsigned char g = (unsigned char)(std::clamp(pixels[i*4+1], 0.0f, 1.0f) * 255.0f);
        unsigned char b = (unsigned char)(std::clamp(pixels[i*4+2], 0.0f, 1.0f) * 255.0f);
        f.write((char*)&r, 1); f.write((char*)&g, 1); f.write((char*)&b, 1);
    }
    std::cout << "Screenshot: " << filename << std::endl;
}
