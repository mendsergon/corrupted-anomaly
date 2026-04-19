// Anomaly orb — main loop. Copies the working display path from blackhole's
// main.cpp: GL 3.3 compatibility profile + glTexSubImage2D + immediate-mode
// fullscreen quad. No FBOs, no core-profile VAO tricks, no shader pipeline --
// GPU writes float RGBA to a buffer, we upload it as a texture, we draw a
// quad. That's the entire display path.

#include "anomaly.h"

#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

// ============================================================================
// Global state
// ============================================================================
static Camera  g_camera;
static int     g_renderWidth  = 1280;
static int     g_renderHeight = 720;
static GLuint  g_texture      = 0;
static int     g_texW = 0, g_texH = 0;

static float   g_simTime  = 0.0f;
static bool    g_paused   = false;
static bool    g_doBloom  = false;  // shape-first: disabled by default; press B to toggle

static bool    g_screenshotReq = false;
static double  g_fps = 0.0;

static const float CAM_SMOOTH = 0.18f;

// ============================================================================
// Callbacks
// ============================================================================

static void framebuffer_size_callback(GLFWwindow*, int w, int h) {
    if (w <= 0 || h <= 0) return;
    glViewport(0, 0, w, h);
    g_renderWidth  = w;
    g_renderHeight = h;
}

static void key_callback(GLFWwindow* win, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(win, GLFW_TRUE); break;
        case GLFW_KEY_SPACE:  if (action == GLFW_PRESS) g_paused  = !g_paused;  break;
        case GLFW_KEY_B:      if (action == GLFW_PRESS) g_doBloom = !g_doBloom; break;
        case GLFW_KEY_R:      if (action == GLFW_PRESS) g_camera  = createDefaultCamera(g_config); break;
        case GLFW_KEY_F12:    if (action == GLFW_PRESS) g_screenshotReq = true; break;
        case GLFW_KEY_EQUAL:  g_camera.targetFov = std::max(15.0f, g_camera.targetFov - 3.0f); break;
        case GLFW_KEY_MINUS:  g_camera.targetFov = std::min(110.0f, g_camera.targetFov + 3.0f); break;
        default: break;
    }
}

static void mouse_button_callback(GLFWwindow* win, int btn, int action, int) {
    if (btn != GLFW_MOUSE_BUTTON_LEFT) return;
    if (action == GLFW_PRESS) {
        g_camera.dragging = true;
        glfwGetCursorPos(win, &g_camera.lastMouseX, &g_camera.lastMouseY);
    } else {
        g_camera.dragging = false;
    }
}

// Rodrigues rotation: rotate v around unit axis a by angle rad.
static cl_float3 rot_axis_vec(cl_float3 v, cl_float3 a, float rad) {
    float c = std::cos(rad), s = std::sin(rad);
    float d = a.s[0]*v.s[0] + a.s[1]*v.s[1] + a.s[2]*v.s[2];
    cl_float3 cr = {
        a.s[1]*v.s[2] - a.s[2]*v.s[1],
        a.s[2]*v.s[0] - a.s[0]*v.s[2],
        a.s[0]*v.s[1] - a.s[1]*v.s[0]
    };
    return {
        v.s[0]*c + cr.s[0]*s + a.s[0]*d*(1.0f - c),
        v.s[1]*c + cr.s[1]*s + a.s[1]*d*(1.0f - c),
        v.s[2]*c + cr.s[2]*s + a.s[2]*d*(1.0f - c)
    };
}

// After applying rotations, re-orthonormalize to kill drift. Treat the
// (possibly rotated) `up` as the most authoritative "horizon" reference,
// recompute `right` from `up x fwd`, then `up` from `fwd x right`.
static void orthonormalize_camera(Camera* cam) {
    float fl = std::sqrt(cam->fwd.s[0]*cam->fwd.s[0] +
                          cam->fwd.s[1]*cam->fwd.s[1] +
                          cam->fwd.s[2]*cam->fwd.s[2]);
    cam->fwd = { cam->fwd.s[0]/fl, cam->fwd.s[1]/fl, cam->fwd.s[2]/fl };

    cl_float3 r = {
        cam->up.s[1]*cam->fwd.s[2] - cam->up.s[2]*cam->fwd.s[1],
        cam->up.s[2]*cam->fwd.s[0] - cam->up.s[0]*cam->fwd.s[2],
        cam->up.s[0]*cam->fwd.s[1] - cam->up.s[1]*cam->fwd.s[0]
    };
    float rl = std::sqrt(r.s[0]*r.s[0] + r.s[1]*r.s[1] + r.s[2]*r.s[2]);
    cam->right = { r.s[0]/rl, r.s[1]/rl, r.s[2]/rl };

    cam->up = {
        cam->fwd.s[1]*cam->right.s[2] - cam->fwd.s[2]*cam->right.s[1],
        cam->fwd.s[2]*cam->right.s[0] - cam->fwd.s[0]*cam->right.s[2],
        cam->fwd.s[0]*cam->right.s[1] - cam->fwd.s[1]*cam->right.s[0]
    };
}

static void cursor_position_callback(GLFWwindow*, double x, double y) {
    if (!g_camera.dragging) return;
    double dx = x - g_camera.lastMouseX;
    double dy = y - g_camera.lastMouseY;
    g_camera.lastMouseX = x;
    g_camera.lastMouseY = y;

    const float s = 0.005f;
    float yaw   = -(float)dx * s;  // rotate around current `up`   (horizontal drag)
    float pitch = -(float)dy * s;  // rotate around current `right` (vertical drag)

    // Yaw first: rotate fwd & right about up. `up` is invariant.
    g_camera.fwd   = rot_axis_vec(g_camera.fwd,   g_camera.up, yaw);
    g_camera.right = rot_axis_vec(g_camera.right, g_camera.up, yaw);

    // Pitch: rotate fwd & up about the updated right. `right` is invariant.
    g_camera.fwd = rot_axis_vec(g_camera.fwd, g_camera.right, pitch);
    g_camera.up  = rot_axis_vec(g_camera.up,  g_camera.right, pitch);

    orthonormalize_camera(&g_camera);
}

static void scroll_callback(GLFWwindow*, double, double yoff) {
    g_camera.targetDistance *= std::pow(0.9f, (float)yoff);
    // Min 0.28 puts the camera inside the hollow membrane with ~0.06 clear
    // of the core surface (R_CORE = 0.22). Max 25 stays unchanged.
    g_camera.targetDistance = std::max(0.28f, std::min(25.0f, g_camera.targetDistance));
}

// ============================================================================
// Texture + fullscreen quad (compat profile immediate mode)
// ============================================================================

static void uploadPixels(const std::vector<float>& pixels, int w, int h) {
    if (g_texture == 0) glGenTextures(1, &g_texture);
    glBindTexture(GL_TEXTURE_2D, g_texture);

    if (w != g_texW || h != g_texH) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, pixels.data());
        g_texW = w; g_texH = h;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, pixels.data());
    }
}

static void drawFullscreenQuad() {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_texture);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f( 1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1,  1);
        glTexCoord2f(0, 1); glVertex2f(-1,  1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    g_config = parseArgs(argc, argv);
    g_renderWidth  = g_config.windowWidth;
    g_renderHeight = g_config.windowHeight;

    std::cout << "============================================\n"
              << "  Corrupted Anomaly — shape pass v2\n"
              << "============================================\n"
              << "  R="        << g_config.shellRadius
              << "  thick="    << g_config.shellThick
              << "  disp="     << g_config.dispAmp
              << "  spike="    << g_config.spikeAmp << "\n"
              << "  holes="    << g_config.holeCount
              << "  hole_r="   << g_config.holeRadius
              << "  drift="    << g_config.holeDrift << "\n"
              << "  core="     << g_config.coreRadius
              << "  veins="    << g_config.veinCount
              << "  vein_r="   << g_config.veinR0 << "->" << g_config.veinR1
              << "  curve="    << g_config.veinCurve << "\n"
              << "  spikes="   << g_config.spikeCount
              << "  spike_h="  << g_config.spikeHeight
              << "  spike_w="  << g_config.spikeWidth
              << "  sharp="    << g_config.spikeSharp << "\n"
              << "  steps="    << g_config.maxSteps << "\n\n";

    if (!glfwInit()) { std::cerr << "glfwInit failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    GLFWwindow* win = glfwCreateWindow(g_config.windowWidth, g_config.windowHeight,
                                       "Anomaly Orb — GPU Raymarcher", nullptr, nullptr);
    if (!win) { std::cerr << "window creation failed\n"; glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(0);
    glfwSetFramebufferSizeCallback(win, framebuffer_size_callback);
    glfwSetKeyCallback           (win, key_callback);
    glfwSetMouseButtonCallback   (win, mouse_button_callback);
    glfwSetCursorPosCallback     (win, cursor_position_callback);
    glfwSetScrollCallback        (win, scroll_callback);

    std::cout << "GL vendor:   " << (const char*)glGetString(GL_VENDOR)   << "\n"
              << "GL renderer: " << (const char*)glGetString(GL_RENDERER) << "\n"
              << "GL version:  " << (const char*)glGetString(GL_VERSION)  << "\n\n";

    if (!g_raymarcher.initialize(g_config)) {
        std::cerr << "GPU raymarcher init failed. Exiting.\n";
        glfwDestroyWindow(win); glfwTerminate();
        return 1;
    }

    g_camera = createDefaultCamera(g_config);

    std::cout << "\nControls:\n"
              << "  LMB + drag   orbit camera\n"
              << "  scroll       zoom\n"
              << "  +/-          FOV\n"
              << "  Space        pause/resume time\n"
              << "  B            toggle bloom\n"
              << "  R            reset camera\n"
              << "  F12          screenshot (PPM)\n"
              << "  Esc          quit\n\n";

    auto t_prev = std::chrono::high_resolution_clock::now();
    double fps_accum = 0.0;
    int    fps_frames = 0;

    while (!glfwWindowShouldClose(win)) {
        auto t_now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(t_now - t_prev).count();
        if (dt > 0.1f) dt = 0.1f;
        t_prev = t_now;

        if (!g_paused) g_simTime += dt;

        g_camera.distance += CAM_SMOOTH * (g_camera.targetDistance - g_camera.distance);
        g_camera.fov      += CAM_SMOOTH * (g_camera.targetFov      - g_camera.fov);

        fps_accum += dt;
        fps_frames++;
        if (fps_accum >= 1.0) {
            g_fps = fps_frames / fps_accum;
            char title[256];
            std::snprintf(title, sizeof(title),
                "Anomaly [%.1f FPS | render %.1fms | r=%.2f fov=%.0f]",
                g_fps, g_raymarcher.getLastRenderMs(),
                g_camera.distance, g_camera.fov);
            glfwSetWindowTitle(win, title);
            fps_accum = 0.0; fps_frames = 0;
        }

        if (g_renderWidth <= 0 || g_renderHeight <= 0) {
            glfwSwapBuffers(win); glfwPollEvents(); continue;
        }

        // Audio stub: simple sine.
        float amp = 0.35f + 0.30f * std::sin(g_simTime * 0.6f)
                          + 0.15f * std::sin(g_simTime * 2.3f);
        if (amp < 0.0f) amp = 0.0f;

        // Render at half res while dragging (matches Kerr pattern), full res otherwise.
        float scale = g_camera.dragging ? 0.5f : 1.0f;
        int rw = std::max(128, (int)(g_renderWidth  * scale));
        int rh = std::max(72,  (int)(g_renderHeight * scale));

        g_raymarcher.renderFrame(g_camera, rw, rh, g_simTime, amp);
        std::vector<float>& pixels = g_raymarcher.getPixels();
        if (pixels.empty() || (int)pixels.size() != rw * rh * 4) {
            glfwSwapBuffers(win); glfwPollEvents(); continue;
        }

        if (g_doBloom && !g_camera.dragging) applyBloom(pixels, rw, rh);

        if (g_screenshotReq) { saveScreenshot(pixels, rw, rh); g_screenshotReq = false; }

        uploadPixels(pixels, rw, rh);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        drawFullscreenQuad();
        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    if (g_texture) glDeleteTextures(1, &g_texture);
    g_raymarcher.cleanup();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
