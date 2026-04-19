// anomaly.cl — Corrupted Arcane Anomaly, shape-first v2.
//
// Changes from v1 (per user feedback):
//  1) Vein endpoints follow the actual displaced inner surface so they
//     don't poke through the shell.
//  2) Hole avoidance: each vein endpoint is rotated along the membrane
//     surface away from any hole whose angular footprint it lands in.
//  3) Web-like veins: 2-segment tapered capsules with a perpendicular
//     midpoint offset for curvature. More fibers, thinner tubes.
//  4) Two-scale displacement: low-freq mountains + high-freq mini-spikes
//     gated by a mid-freq mask, so some regions grow dense fine spikes
//     while others stay smooth.
//  5) Bigger, sparser holes by default.
//
// A per-thread SceneCtx holds precomputed hole centers/radii and vein
// endpoints. build_ctx() runs once at the top of the kernel; map_scene()
// then only reads from ctx, so the sphere-tracer doesn't recompute any
// of the per-frame invariants.

#ifndef R_SHELL
#define R_SHELL       1.0f
#endif
#ifndef THICK
#define THICK         0.04f
#endif
#ifndef DISP_AMP
#define DISP_AMP      0.12f     // low-freq mountain amplitude
#endif
#ifndef SPIKE_AMP
#define SPIKE_AMP     0.06f     // high-freq mini-spike amplitude (gated)
#endif
#ifndef R_HOLE
#define R_HOLE        0.30f
#endif
#ifndef N_HOLES
#define N_HOLES       24
#endif
#ifndef HOLE_DRIFT
#define HOLE_DRIFT    0.10f
#endif
#ifndef N_REGIONS
#define N_REGIONS     8
#endif
#ifndef HOLE_COV_FRAC
#define HOLE_COV_FRAC 0.33f
#endif
#ifndef R_CORE
#define R_CORE        0.22f
#endif
#ifndef N_VEINS
#define N_VEINS       48
#endif
#ifndef VEIN_R0
#define VEIN_R0       0.018f
#endif
#ifndef VEIN_R1
#define VEIN_R1       0.006f
#endif
#ifndef VEIN_CURVE
#define VEIN_CURVE    0.18f     // perpendicular midpoint offset magnitude
#endif
// Regional allocation for membrane attachments -- analog of the 8-region
// Voronoi system the holes use. Each vein is hard-assigned to one of
// N_REGIONS fib-distributed direction centers; within each region, the
// N_VEINS/N_REGIONS veins spread in a flower pattern via distinct
// tangent-plane offsets of varying magnitude.
#ifndef N_REGIONS
#define N_REGIONS     8
#endif
#ifndef VEIN_SPREAD
#define VEIN_SPREAD   0.22f     // max angular spread per region (~13 deg)
#endif
// --- Web topology additions -----------------------------------------------
// Core hubs: visible sphere nodes where veins converge at the core.
#ifndef N_HUBS_C
#define N_HUBS_C      6
#endif
#ifndef HUB_R_C
#define HUB_R_C       0.032f
#endif
// Bridges: thin cross-connection capsules between pairs of vein midpoints.
// Turns the bundle of trunks into a graph.
#ifndef N_BRIDGES
#define N_BRIDGES     20
#endif
#ifndef BRIDGE_R
#define BRIDGE_R      0.007f
#endif
// Relative radius modulation along each vein's length; gives organic bumps
// instead of smooth cones. Bounded by sphere-trace safety factor.
#ifndef VEIN_R_NOISE
#define VEIN_R_NOISE  0.30f
#endif
// Final interior/membrane smin -- controls how broadly hubs/veins open up
// into the inner membrane surface.
#ifndef FUSE_K
#define FUSE_K        0.030f
#endif
// Cardiac / flow ------------------------------------------------------------
// HEART_RATE sets both the core beat frequency and the blood-flow wave speed
// along veins (one wave traverses a vein per beat -- phase-locked so the
// vein pulse appears to emanate from the contracting core).
#ifndef HEART_RATE
#define HEART_RATE       0.60f  // beats per second (~36 bpm, slow-dramatic)
#endif
#ifndef HEART_PUMP_AMP
#define HEART_PUMP_AMP   0.12f  // fractional core radius swing at peak systole
#endif
#ifndef VEIN_PULSE_AMP
#define VEIN_PULSE_AMP   0.18f  // fractional vein radius bulge at wave crest
#endif
#ifndef N_SPIKES
#define N_SPIKES      96
#endif
#ifndef N_SMALL
#define N_SMALL       32
#endif
#ifndef SPIKE_H
#define SPIKE_H       0.36f     // max outward protrusion of a spike at its peak
#endif
#ifndef SPIKE_W
#define SPIKE_W       0.14f     // spike base half-angle in radians (~8 deg)
#endif
#ifndef SPIKE_SHARP
#define SPIKE_SHARP   5.5f      // cone profile exponent; higher = pointier
#endif
#ifndef SPIKE_DRIFT
#define SPIKE_DRIFT   0.12f     // slow per-spike axis rotation rate
#endif
#ifndef SPIKE_RATE
#define SPIKE_RATE    0.35f     // per-spike height oscillation frequency
#endif

// Partition of the N_SPIKES slots:
//   [0,            N_SOLO)        solo big spikes with wide size variance
//   [N_SOLO,       N_MAIN)        cluster members (N_CLUSTERS groups of 4)
//   [N_MAIN,       N_SPIKES)      small scattered spikes
// N_MAIN = solo + cluster share of the budget; the remainder is small.
#define CLUSTER_SIZE     4
#define N_MAIN           (((N_SPIKES) > (N_SMALL)) ? ((N_SPIKES) - (N_SMALL)) : 0)
#define N_CLUSTERS       ((N_MAIN) / 16)
#define N_SOLO           ((N_MAIN) - (N_CLUSTERS) * (CLUSTER_SIZE))
#define N_CLUSTERS_ALLOC (((N_CLUSTERS) > 0) ? (N_CLUSTERS) : 1)

#ifndef MAX_STEPS
#define MAX_STEPS     128
#endif

#define PI           3.14159265359f
#define GOLDEN_ANGLE 2.39996323f

// ============================================================================
// Noise
// ============================================================================
static float fract1(float x) { return x - floor(x); }

static float3 hash3(float3 p) {
    p = (float3)(dot(p, (float3)(127.1f, 311.7f,  74.7f)),
                 dot(p, (float3)(269.5f, 183.3f, 246.1f)),
                 dot(p, (float3)(113.5f, 271.9f, 124.6f)));
    return (float3)(fract1(native_sin(p.x) * 43758.5453f) - 0.5f,
                    fract1(native_sin(p.y) * 43758.5453f) - 0.5f,
                    fract1(native_sin(p.z) * 43758.5453f) - 0.5f) * 2.0f;
}

static float vnoise3(float3 p) {
    float3 i = floor(p);
    float3 f = p - i;
    float3 u = f * f * (3.0f - 2.0f * f);

    float3 d0 = (float3)(0, 0, 0), d1 = (float3)(1, 0, 0),
           d2 = (float3)(0, 1, 0), d3 = (float3)(1, 1, 0),
           d4 = (float3)(0, 0, 1), d5 = (float3)(1, 0, 1),
           d6 = (float3)(0, 1, 1), d7 = (float3)(1, 1, 1);

    float n000 = dot(hash3(i + d0), f - d0);
    float n100 = dot(hash3(i + d1), f - d1);
    float n010 = dot(hash3(i + d2), f - d2);
    float n110 = dot(hash3(i + d3), f - d3);
    float n001 = dot(hash3(i + d4), f - d4);
    float n101 = dot(hash3(i + d5), f - d5);
    float n011 = dot(hash3(i + d6), f - d6);
    float n111 = dot(hash3(i + d7), f - d7);

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

static float fbm3(float3 p) {
    float a = 0.5f, s = 0.0f;
    float3 q = p;
    for (int i = 0; i < 3; i++) {
        s += a * vnoise3(q);
        q *= 2.03f;
        a *= 0.5f;
    }
    return s;
}

// Low-freq mountain displacement only (used by veins for surface-following
// endpoints -- veins attach to a clean inner membrane, not to spike tips).
static float mountain_disp(float3 dir, float t) {
    float3 q = dir * 2.5f + (float3)(t * 0.15f, t * 0.10f, -t * 0.12f);
    return fbm3(q) * DISP_AMP;
}

// The mid-freq "roughness" mask. Positive-valued regions grow mini-spike
// texture on the shell AND are where the discrete aggressive spikes can
// emerge; negative regions stay smooth.
static float rough_mask(float3 dir, float t) {
    float3 q = dir * 1.8f + (float3)(t * 0.07f, -t * 0.05f, t * 0.08f);
    return fbm3(q);
}

// Inner surface displacement: mountains + mini-spike texture gated by the
// roughness mask. No big spikes. This is what the veins track.
static float inner_disp(float3 dir, float t) {
    float m     = mountain_disp(dir, t);
    float mask  = rough_mask(dir, t);
    float gate  = smoothstep(0.02f, 0.25f, mask);
    float3 qhi  = dir * 11.0f + (float3)(t * 0.35f, t * 0.30f, -t * 0.40f);
    float  hi   = fbm3(qhi);
    return m + hi * gate * SPIKE_AMP;
}

// ============================================================================
// SDF primitives
// ============================================================================
static float sd_taper(float3 p, float3 a, float3 b, float ra, float rb) {
    float3 pa = p - a;
    float3 ba = b - a;
    float L2 = dot(ba, ba);
    float h  = clamp(dot(pa, ba) / L2, 0.0f, 1.0f);
    float r  = mix(ra, rb, h);
    return length(pa - ba * h) - r;
}

// Constant-radius capsule for bridges (cross-connections).
static float sd_capsule(float3 p, float3 a, float3 b, float r) {
    float3 pa = p - a;
    float3 ba = b - a;
    float L2 = dot(ba, ba);
    float h  = clamp(dot(pa, ba) / L2, 0.0f, 1.0f);
    return length(pa - ba * h) - r;
}

// Two-frequency 1D noise in ~[-0.85, +0.85], seeded so different veins get
// different bump patterns.
static float noise1d_seed(float h, float seed) {
    float x = h * 4.5f + seed * 6.2831853f;
    return native_sin(x) * 0.5f
         + native_sin(x * 2.31f + seed * 1.7f) * 0.35f;
}

// Cardiac pump waveform. Phase phi = frac(t * HEART_RATE); returns a near-zero
// baseline through diastole, a sharp primary peak at phi=0.08 (S1, main
// contraction), and a softer secondary peak at phi=0.35 (S2). Peak total
// ~1.0 at S1. Two gaussians summed.
static float heart_pump(float t) {
    float phi = t * HEART_RATE;
    phi = phi - floor(phi);
    float s1 = (phi - 0.08f) / 0.04f;
    float s2 = (phi - 0.35f) / 0.06f;
    return native_exp(-s1 * s1) + 0.35f * native_exp(-s2 * s2);
}

// Tapered capsule with radius modulation along length. The reported SDF
// stops being a true Euclidean distance (it's a bound) once radius varies,
// but the bound is tight enough that sphere-tracing with the existing 0.50
// safety factor converges.
//
// h_local in [0, 1] runs across this segment; h_global = seg_start +
// h_local * seg_extent is the 0..1 parameter spanning the full vein, used
// for the traveling flow pulse so it crosses segment boundaries without
// phase discontinuity. Static radius noise stays segment-local (per-segment
// seed), since there's no reason for bumps to align at joints.
static float sd_taper_noisy(float3 p, float3 a, float3 b,
                            float ra, float rb, float seed,
                            float t_sim, float seg_start, float seg_extent) {
    float3 pa = p - a;
    float3 ba = b - a;
    float L2 = dot(ba, ba);
    float h  = clamp(dot(pa, ba) / L2, 0.0f, 1.0f);
    float r_base = mix(ra, rb, h);

    float h_global = seg_start + h * seg_extent;
    const float TAU = 6.2831853f;
    float pulse = (native_cos(TAU * (h_global - t_sim * HEART_RATE)) + 1.0f) * 0.5f;

    float r = r_base * (1.0f
                        + VEIN_R_NOISE   * noise1d_seed(h, seed)
                        + VEIN_PULSE_AMP * pulse);
    return length(pa - ba * h) - r;
}

static float smin_poly(float a, float b, float k) {
    float h = clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return mix(b, a, h) - k * h * (1.0f - h);
}

// Flowy vein: a -> m1 -> m2 -> m3 -> b, four tapered segments smin-joined
// at every interior control point. With the three control points placed
// non-coplanarly (see createCtx), the resulting curve is a proper 3D space
// curve -- S-shapes, helical twists, figure-8 hints -- rather than the
// planar arc you get with a single mid-kink. Per-segment seed offsets so
// the radius noise doesn't align across joins; per-segment (seg_start,
// seg_extent) so the flow pulse's h parameter is continuous across the
// whole vein.
static float sd_curved_vein(float3 p,
                            float3 a, float3 m1, float3 m2, float3 m3, float3 b,
                            float r0, float r1, float seed, float t_sim) {
    float r_m1 = mix(r0, r1, 0.25f);
    float r_m2 = mix(r0, r1, 0.50f);
    float r_m3 = mix(r0, r1, 0.75f);

    float d1 = sd_taper_noisy(p, a,  m1, r0,   r_m1, seed,          t_sim, 0.00f, 0.25f);
    float d2 = sd_taper_noisy(p, m1, m2, r_m1, r_m2, seed + 1.17f,  t_sim, 0.25f, 0.25f);
    float d3 = sd_taper_noisy(p, m2, m3, r_m2, r_m3, seed + 2.41f,  t_sim, 0.50f, 0.25f);
    float d4 = sd_taper_noisy(p, m3, b,  r_m3, r1,   seed + 3.67f,  t_sim, 0.75f, 0.25f);

    float d = smin_poly(d1, d2, 0.008f);
    d       = smin_poly(d,  d3, 0.008f);
    d       = smin_poly(d,  d4, 0.008f);
    return d;
}

// ============================================================================
// Ctx construction
// ============================================================================
static float3 fib_dir(int i, int n) {
    float z   = 1.0f - (2.0f * (float)i + 1.0f) / (float)n;
    float r   = native_sqrt(fmax(1.0f - z * z, 0.0f));
    float phi = (float)i * GOLDEN_ANGLE;
    return (float3)(r * native_cos(phi), z, r * native_sin(phi));
}

static float3 rot_axis(float3 v, float3 axis, float a) {
    float c = native_cos(a), s = native_sin(a);
    return v * c + cross(axis, v) * s + axis * (dot(axis, v) * (1.0f - c));
}

// Pick any unit vector perpendicular to `ref`.
static float3 any_perp(float3 ref) {
    float3 h = (fabs(ref.y) < 0.9f) ? (float3)(0.0f, 1.0f, 0.0f)
                                    : (float3)(1.0f, 0.0f, 0.0f);
    return normalize(cross(ref, h));
}

// Unit tangent at v that points away from ref (on the great circle through v
// and ref). Falls back to any_perp when v is parallel to ref.
static float3 tangent_away(float3 v, float3 ref) {
    float c = dot(v, ref);
    float3 t = v - ref * c;
    float tl = length(t);
    if (tl > 1e-4f) return t / tl;
    return any_perp(ref);
}

typedef struct {
    float3 hole_c[N_HOLES];
    float  hole_r[N_HOLES];
    float  hole_cos_b[N_HOLES];     // cos(asin(r/R_SHELL)); angular-radius cosine
    float3 hole_pk1[N_HOLES];       // hole tangent-plane basis, vector 1
    float3 hole_pk2[N_HOLES];       // hole tangent-plane basis, vector 2
    float  hole_pp1[N_HOLES];       // phase for the u-direction perturbation
    float  hole_pp2[N_HOLES];       // phase for the v-direction perturbation
    float3 vein_a[N_VEINS];
    float3 vein_m1[N_VEINS];        // interior control at t=0.25
    float3 vein_m2[N_VEINS];        // interior control at t=0.50
    float3 vein_m3[N_VEINS];        // interior control at t=0.75
    float3 vein_b[N_VEINS];
    float  vein_seed[N_VEINS];      // per-vein noise seed for radius bumps
    float3 hub_p[N_HUBS_C];         // core-side junction nodes (visible spheres)
    float  hub_r[N_HUBS_C];
    float3 bridge_a[N_BRIDGES];     // cross-connection endpoints
    float3 bridge_b[N_BRIDGES];
    float3 spike_c[N_SPIKES];       // unit direction to each spike
    float  spike_h[N_SPIKES];       // current height (0 when suppressed)
    float  spike_cos_w[N_SPIKES];   // per-spike precomputed cos(width_i)
    float  spike_inv_ocw[N_SPIKES]; // per-spike 1 / (1 - cos(width_i))
    float  t;
} SceneCtx;

// Nudge a unit direction `d` out of every hole it currently overlaps by
// snapping it onto the worst offender's angular boundary. Discrete and
// effective, but the identity of the worst offender changes discontinuously
// as holes drift, so the output has snap-steps -- fine for static or
// self-drifting directions (spikes), NOT fine for vein endpoints where the
// snap reads as a teleport across the gap. Use smooth_escape for those.
static float3 escape_holes(float3 d, __private const float3* hole_dir,
                           __private const float* hole_ang) {
    for (int iter = 0; iter < 3; iter++) {
        int   worst = -1;
        float worst_slack = 0.0f;  // positive = still inside some hole
        for (int h = 0; h < N_HOLES; h++) {
            float c = dot(d, hole_dir[h]);
            float cos_safe = native_cos(hole_ang[h]);
            float slack    = c - cos_safe;
            if (slack > worst_slack) { worst_slack = slack; worst = h; }
        }
        if (worst < 0) break;

        float3 hd = hole_dir[worst];
        float3 tn = tangent_away(d, hd);
        float  ang = hole_ang[worst];
        d = hd * native_cos(ang) + tn * native_sin(ang);
        d = normalize(d);
    }
    return d;
}

// Continuous version of escape_holes. Each hole contributes a tangent push
// whose magnitude is a smooth (smoothstep) function of how deep d is inside
// the hole's angular cone. Because the magnitude is C^1 in the dot product,
// and dot is smooth in hole_dir, the output d' is a smooth function of the
// hole positions -- so as a hole drifts toward the hub, the hub slides out
// of the way continuously rather than teleporting when some "worst violator"
// changes identity. Three passes give enough cumulative push to clear even
// centered overlaps; each pass is smooth, so their composition is smooth.
static float3 smooth_escape(float3 d, __private const float3* hole_dir,
                            __private const float* hole_ang) {
    for (int pass = 0; pass < 3; pass++) {
        float3 push = (float3)(0.0f, 0.0f, 0.0f);
        for (int h = 0; h < N_HOLES; h++) {
            float c        = dot(d, hole_dir[h]);
            float cos_safe = native_cos(hole_ang[h]);
            // smoothstep from "just outside safe boundary" to "well inside"
            float overlap = smoothstep(cos_safe - 0.12f, cos_safe + 0.02f, c);
            if (overlap > 0.0f) {
                // Tangent at d away from the hole center.
                float3 tproj = d - hole_dir[h] * c;
                float  tl    = length(tproj);
                float3 pd    = (tl > 1e-4f) ? tproj / tl : any_perp(hole_dir[h]);
                push += pd * overlap * 0.20f;  // max 0.20 rad per-hole per-pass
            }
        }
        d = normalize(d + push);
    }
    return d;
}

// Continuous arc-visibility adjustment: rotates d_target around d_region so
// the great-circle arc clears hole cones, with motion that's a smooth
// function of the hole positions. The degree of freedom is rotation of
// d_target around d_region (keeps arc length constant, rotates the arc
// plane). Push direction r_hat = normalize(cross(d_region, d_target)) is
// the tangent on the sphere along which that rotation moves d_target; push
// magnitude is -dot(r_hat, hole_dir) * overlap (gradient of penetration
// w.r.t. rotation angle, times smoothstep of penetration). Every operation
// is C^1 in hole_dir, so no discrete switches -> no teleports.
static float3 smooth_arc_escape(float3 d_region, float3 d_target,
                                __private const float3* hole_dir,
                                __private const float* hole_ang) {
    for (int pass = 0; pass < 3; pass++) {
        float c_ab = dot(d_region, d_target);
        if (c_ab > 0.9995f) break;  // arc is essentially zero-length
        float theta = acos(clamp(c_ab, -0.9999f, 0.9999f));

        float3 t_arc = d_target - d_region * c_ab;
        float  tl    = length(t_arc);
        if (tl < 1e-4f) break;
        t_arc /= tl;

        // r_hat: unit tangent at d_target along the circle of rotation
        // around d_region. |cross(d_region, d_target)| = sin(theta), which
        // is non-zero here since c_ab < 0.9995.
        float3 rraw = cross(d_region, d_target);
        float  rl   = length(rraw);
        if (rl < 1e-4f) break;
        float3 r_hat = rraw / rl;

        float3 push = (float3)(0.0f, 0.0f, 0.0f);
        for (int h = 0; h < N_HOLES; h++) {
            float c_r      = dot(d_region, hole_dir[h]);
            float c_t      = dot(t_arc,    hole_dir[h]);
            float c_end    = dot(d_target, hole_dir[h]);
            float cos_safe = native_cos(hole_ang[h]);

            // Closed-form max of penetration along the arc.
            float a_star = atan2(c_t, c_r);
            float f_max;
            if (a_star >= 0.0f && a_star <= theta) {
                f_max = native_sqrt(c_r * c_r + c_t * c_t);
            } else {
                f_max = fmax(c_r, c_end);
            }

            float overlap = smoothstep(cos_safe - 0.12f, cos_safe + 0.02f, f_max);
            if (overlap <= 0.0f) continue;

            // Gradient of penetration w.r.t. rotation angle at d_target is
            // dot(r_hat, hole_dir); pushing in -r_hat*grad*overlap moves
            // d_target in the direction that reduces penetration.
            float grad = dot(r_hat, hole_dir[h]);
            push += -r_hat * (grad * overlap * 0.30f);
        }

        d_target = normalize(d_target + push);
    }
    return d_target;
}

static void build_ctx(SceneCtx* ctx, float t) {
    ctx->t = t;

    // Holes ---------------------------------------------------------------
    // Each hole carries (position, radius, tangent-frame basis, phases).
    // The tangent-frame (pk1, pk2) is orthonormal to the hole direction
    // and rotates *with* the hole as it drifts -- all three are obtained
    // by applying the same rot_axis to a base (base_dir, base_pk1, base_pk2)
    // triple. This keeps the boundary-noise pattern rigid on the hole
    // (no morphing while it moves, no discontinuity at any_perp's switch
    // point) while still smoothly varying across frames.
    float3 hole_dir[N_HOLES];
    float  hole_ang[N_HOLES];
    for (int i = 0; i < N_HOLES; i++) {
        float3 base_dir = fib_dir(i, N_HOLES);
        float3 base_pk1 = any_perp(base_dir);
        float3 base_pk2 = cross(base_dir, base_pk1);

        float3 axis_raw = (float3)(0.3f + 0.2f * (float)i,
                                    1.0f,
                                   -0.1f + 0.15f * (float)i);
        float3 axis = normalize(axis_raw);

        // Per-hole speed via hash: keeps the range tight (0.7 .. 1.3 x base)
        // so high-index holes no longer run away linearly with i.
        float hspd  = fract1(native_sin((float)i * 5.13f + 2.71f) * 12345.678f);
        float speed = HOLE_DRIFT * (0.7f + 0.6f * hspd);
        float angle = t * speed;

        ctx->hole_c[i]   = rot_axis(base_dir, axis, angle) * R_SHELL;
        ctx->hole_pk1[i] = rot_axis(base_pk1, axis, angle);
        ctx->hole_pk2[i] = rot_axis(base_pk2, axis, angle);
        hole_dir[i]      = rot_axis(base_dir, axis, angle);

        // Size: power-shaped variance + slower, lower-amplitude wobble.
        float h_raw  = fract1(native_sin((float)i * 7.123f + 3.456f) * 98765.4321f);
        float size_f = 0.30f + 1.60f * h_raw * h_raw;
        ctx->hole_r[i] = R_HOLE * size_f *
                         (0.90f + 0.10f * native_sin(t * 0.18f + (float)i * 2.1f));

        // Per-hole boundary perturbation phases (fixed; shape stays consistent).
        ctx->hole_pp1[i] = (float)i * 3.30f;
        ctx->hole_pp2[i] = (float)i * 5.70f + 1.41f;
    }

    // Regional coverage cap with SOFT assignment via softmax. Each hole
    // contributes its solid-angle footprint to every region weighted by
    // exp(BETA * dot(hole_dir, region_dir)) / sum, then each hole's
    // radius scale is the weighted average of the per-region scales.
    // Smooths the discontinuous snaps of nearest-neighbor assignment.
    float3 region_dir[N_REGIONS];
    for (int r = 0; r < N_REGIONS; r++) {
        region_dir[r] = fib_dir(r, N_REGIONS);
    }

    float region_cov[N_REGIONS];
    for (int r = 0; r < N_REGIONS; r++) region_cov[r] = 0.0f;

    const float BETA = 8.0f;

    // Pass 1: softmax-weighted coverage accumulation
    for (int i = 0; i < N_HOLES; i++) {
        float exps[N_REGIONS];
        float sum_exp = 0.0f;
        for (int r = 0; r < N_REGIONS; r++) {
            float c = dot(hole_dir[i], region_dir[r]);
            exps[r] = native_exp(BETA * c);
            sum_exp += exps[r];
        }
        float inv = 1.0f / sum_exp;

        float r_rel  = clamp(ctx->hole_r[i] / R_SHELL, 0.0f, 0.98f);
        float cos_a  = native_sqrt(1.0f - r_rel * r_rel);
        float omega  = 2.0f * PI * (1.0f - cos_a);

        for (int r = 0; r < N_REGIONS; r++) {
            region_cov[r] += omega * exps[r] * inv;
        }
    }

    // Per-region required scale factor (1.0 if under budget).
    float region_scale[N_REGIONS];
    const float REGION_AREA = 4.0f * PI / (float)N_REGIONS;
    const float MAX_COV     = REGION_AREA * HOLE_COV_FRAC;
    for (int r = 0; r < N_REGIONS; r++) {
        region_scale[r] = (region_cov[r] > MAX_COV)
                          ? native_sqrt(MAX_COV / region_cov[r])
                          : 1.0f;
    }

    // Pass 2: each hole gets the softmax-weighted average of region scales.
    for (int i = 0; i < N_HOLES; i++) {
        float exps[N_REGIONS];
        float sum_exp = 0.0f;
        for (int r = 0; r < N_REGIONS; r++) {
            float c = dot(hole_dir[i], region_dir[r]);
            exps[r] = native_exp(BETA * c);
            sum_exp += exps[r];
        }
        float inv = 1.0f / sum_exp;

        float scale = 0.0f;
        for (int r = 0; r < N_REGIONS; r++) {
            scale += exps[r] * inv * region_scale[r];
        }
        ctx->hole_r[i] *= scale;
    }

    // Finalize angular radii + safety margin from the (possibly scaled) r.
    // Also precompute cos(angular_radius) for thickness modulation.
    for (int i = 0; i < N_HOLES; i++) {
        float r_rel = clamp(ctx->hole_r[i] / R_SHELL, 0.0f, 0.98f);
        ctx->hole_cos_b[i] = native_sqrt(1.0f - r_rel * r_rel);
        float a = asin(r_rel);
        hole_ang[i] = a + 0.09f;
    }

    // Hubs ----------------------------------------------------------------
    // N_HUBS_C nodes on the core surface, N_HUBS_M nodes on the inner
    // membrane surface (tracking its displacement, hole-escaped). Veins
    // attach to these so multiple fibers converge at each node instead of
    // flying to random points on the inner wall.
    for (int k = 0; k < N_HUBS_C; k++) {
        // Offset + stride so directions aren't trivially axis-aligned.
        float3 d = fib_dir(k * 2 + 1, N_HUBS_C * 2 + 1);
        ctx->hub_p[k] = d * (R_CORE + 0.012f);
        float jitter  = fract1(native_sin((float)k * 1.73f + 0.31f) * 543.21f);
        ctx->hub_r[k] = HUB_R_C * (0.80f + 0.45f * jitter);
    }

    // Veins ---------------------------------------------------------------
    // Core side: all veins sharing a core hub attach exactly at that hub's
    // center -- N_VEINS/N_HUBS_C = 8 veins per hub emerge like anemone arms.
    //
    // Membrane side: regional allocation analogous to the 8-region hole
    // Voronoi. Each vein is hard-assigned to one of N_REGIONS fib-distributed
    // region centers; the N_VEINS/N_REGIONS = 6 veins in each region spread
    // into a flower pattern via distinct tangent-plane offsets of varying
    // magnitude (up to VEIN_SPREAD radians). This gives clustered-but-not-
    // coincident attachments: groups of veins sprouting from a common region
    // of the membrane, each with its own attach point.
    for (int i = 0; i < N_VEINS; i++) {
        int ch = i % N_HUBS_C;               // core hub (8 veins each)
        int r  = i % N_REGIONS;              // membrane region
        int l  = i / N_REGIONS;              // local index within region

        ctx->vein_a[i] = ctx->hub_p[ch];

        // Compute membrane endpoint direction -----------------------------
        float3 d_region = fib_dir(r, N_REGIONS);

        // Tangent offset direction unique per (region, local) pair. Using a
        // scrambled seed so neighbouring local indices don't all land in the
        // same direction.
        int   seed_i = l * 13 + r * 7 + 5;
        float3 raw   = fib_dir(seed_i % 31 + 3, 41);
        float3 tang  = raw - d_region * dot(raw, d_region);
        float  tl    = length(tang);
        if (tl < 1e-4f) tang = any_perp(d_region); else tang /= tl;

        // Per-vein spread magnitude: 40% to 100% of VEIN_SPREAD so the six
        // veins in each region sit at varying angular distances from the
        // region center rather than all on the same ring.
        float spread_frac = 0.40f + 0.60f *
                            fract1(native_sin((float)seed_i * 4.37f) * 12345.0f);
        float  spread = VEIN_SPREAD * spread_frac;

        // Build the desired endpoint, then apply two smooth-gradient passes:
        //  1. smooth_arc_escape: rotates d_vein around d_region so the
        //     great-circle arc clears hole cones (spherical visibility
        //     adjustment, continuous in hole positions).
        //  2. smooth_escape: safety net, pushes the endpoint itself out of
        //     any hole in case the arc adjustment couldn't fully clear.
        // Both are C^1 in the hole positions -- no discrete jumps.
        float3 d_vein = d_region * native_cos(spread) + tang * native_sin(spread);
        d_vein = smooth_arc_escape(d_region, d_vein, hole_dir, hole_ang);
        d_vein = smooth_escape(d_vein, hole_dir, hole_ang);

        // Apply dampened membrane displacement and recess slightly so the
        // vein tip fuses into the inner surface via the final smin.
        float disp    = mountain_disp(d_vein, t) * 0.60f;
        float R_inner = R_SHELL - THICK * 0.5f + disp - 0.010f;
        ctx->vein_b[i] = d_vein * R_inner;

        // Build two perpendicular-to-line bend axes (perp1, perp2). The
        // first is derived from a fib index so neighbouring veins curve
        // differently; the second is Gram-Schmidt'd off the first in the
        // plane perpendicular to the line, so it's genuinely a second
        // curvature axis (not a copy of perp1). With two perpendicular
        // axes the control points live on a 3D space curve, not a plane.
        float3 line = ctx->vein_b[i] - ctx->vein_a[i];
        float  Ll   = length(line);
        float3 lu   = (Ll > 1e-4f) ? (line / Ll) : (float3)(0.0f, 1.0f, 0.0f);

        float3 raw1 = fib_dir((i * 13 + 2) % N_VEINS, N_VEINS);
        float3 perp1 = raw1 - lu * dot(raw1, lu);
        float  l1    = length(perp1);
        if (l1 < 1e-4f) perp1 = any_perp(lu); else perp1 /= l1;

        float3 raw2  = fib_dir((i * 7 + 5) % N_VEINS, N_VEINS);
        float3 perp2 = raw2 - lu * dot(raw2, lu);         // remove line component
        perp2        = perp2 - perp1 * dot(perp2, perp1); // remove perp1 component
        float  l2    = length(perp2);
        if (l2 < 1e-4f) perp2 = normalize(cross(lu, perp1));
        else            perp2 /= l2;

        // Per-vein phase for the secondary S-wave so each vein twists
        // differently. 2.39 is just "doesn't divide 2*pi".
        float phase_i = (float)i * 2.39f;

        // Place three interior control points along the line, each offset
        // by two perpendicular components:
        //   primary:   sin(pi * t)          -- arc bulge, peaks at midpoint
        //   secondary: sin(2*pi * t + phi)  -- S-wave, sign varies along length
        // The secondary amplitude (0.60 of primary) keeps it a supporting
        // wiggle rather than dominating the trajectory.
        const float TAU = 6.2831853f;
        const float C   = VEIN_CURVE;
        const float Cs  = VEIN_CURVE * 0.60f;

        float3 ab = ctx->vein_b[i] - ctx->vein_a[i];

        float A1 = native_sin(PI * 0.25f) * C;
        float A2 = native_sin(PI * 0.50f) * C;
        float A3 = native_sin(PI * 0.75f) * C;

        float B1 = native_sin(TAU * 0.25f + phase_i) * Cs;
        float B2 = native_sin(TAU * 0.50f + phase_i) * Cs;
        float B3 = native_sin(TAU * 0.75f + phase_i) * Cs;

        ctx->vein_m1[i] = ctx->vein_a[i] + ab * 0.25f + perp1 * A1 + perp2 * B1;
        ctx->vein_m2[i] = ctx->vein_a[i] + ab * 0.50f + perp1 * A2 + perp2 * B2;
        ctx->vein_m3[i] = ctx->vein_a[i] + ab * 0.75f + perp1 * A3 + perp2 * B3;

        ctx->vein_seed[i] = (float)i * 0.7374f + 0.3183f;
    }

    // Bridges -------------------------------------------------------------
    // Thin cross-connections between pairs of vein middle-control points
    // (m2 = the sin(pi/2) peak, the natural "waist" of each vein). Strides
    // coprime with N_VEINS so bridges aren't neighbour-clumped.
    {
        const int offsets[4] = { 3, 5, 7, 11 };
        for (int k = 0; k < N_BRIDGES; k++) {
            int off = offsets[k & 3];
            int i = k % N_VEINS;
            int j = (i + off) % N_VEINS;
            ctx->bridge_a[k] = ctx->vein_m2[i];
            ctx->bridge_b[k] = ctx->vein_m2[j];
        }
    }

    // Spike cluster seeds -------------------------------------------------
    // A cluster has a shared seed direction and all its members offset from
    // it by a small angle. Seeds drift slowly (at 0.35 x SPIKE_DRIFT, same
    // rate regardless of cluster) so each cluster moves as one clump.
    float3 cluster_seeds[N_CLUSTERS_ALLOC];
    for (int c = 0; c < N_CLUSTERS; c++) {
        float3 seed_base = fib_dir(c * 3 + 7, N_CLUSTERS * 3 + 7);
        float3 seed_axis = normalize((float3)(0.5f + 0.10f * (float)c,
                                              -0.3f,
                                               1.0f - 0.15f * (float)c));
        float  seed_speed = SPIKE_DRIFT * 0.35f;
        float3 seed_dir = rot_axis(seed_base, seed_axis, t * seed_speed);
        seed_dir = escape_holes(seed_dir, hole_dir, hole_ang);
        cluster_seeds[c] = seed_dir;
    }

    // Spikes --------------------------------------------------------------
    // Three categories packed into one array:
    //   [0, N_SOLO)      : solo big spikes, wide size variance
    //   [N_SOLO, N_MAIN) : cluster members, packed around shared seeds
    //   [N_MAIN, N_SPIKES): small scattered spikes, their own Fib lattice
    // Height oscillation rate is `SPIKE_RATE / (0.5 + 0.5·size)`, so small
    // spikes naturally pulse faster than big ones -- no special handling.
    for (int i = 0; i < N_SPIKES; i++) {
        float  size;
        float3 dir;

        if (i < N_SOLO) {
            float h_raw = fract1(native_sin((float)i * 12.9898f + 78.233f) * 43758.5453f);
            size = 0.50f + 2.50f * h_raw * h_raw;   // [0.50, 3.00]

            float3 base = fib_dir(i * 2 + 1, N_SOLO * 2);
            float3 axis = normalize((float3)(-0.2f + 0.13f * (float)i,
                                              1.0f,
                                              0.35f - 0.11f * (float)i));
            float speed = SPIKE_DRIFT / (0.4f + size);
            dir = rot_axis(base, axis, t * speed);
        } else if (i < N_MAIN) {
            int k   = i - N_SOLO;
            int cid = k / CLUSTER_SIZE;
            int mid = k % CLUSTER_SIZE;

            float3 seed_dir = cluster_seeds[cid];
            float  member_h = fract1(native_sin((float)i * 21.37f + 11.1f) * 98765.43f);

            if (mid == 0) {
                dir  = seed_dir;
                size = 0.70f + 0.25f * member_h;     // [0.70, 0.95]
            } else {
                float3 perp = any_perp(seed_dir);
                float  phi  = ((float)(mid - 1)) * (2.0f * PI / (float)(CLUSTER_SIZE - 1))
                            + 0.3f * (float)cid;
                perp = rot_axis(perp, seed_dir, phi);
                float offset_ang = 0.09f + 0.04f * member_h;
                dir  = rot_axis(seed_dir, perp, offset_ang);
                size = 0.35f + 0.35f * member_h;     // [0.35, 0.70]
            }
        } else {
            // Small scattered spike
            int s = i - N_MAIN;
            float h_raw = fract1(native_sin((float)s * 17.531f + 91.127f) * 54321.9876f);
            size = 0.10f + 0.25f * h_raw;            // [0.10, 0.35]

            // Own Fib lattice on N_SMALL points, offset by +1 to start at
            // a mid-latitude point so no small spike sits exactly on a pole
            // where normalize-perp is numerically unfriendly.
            float3 base = fib_dir(s + 1, N_SMALL + 2);
            float3 axis = normalize((float3)( 0.4f + 0.10f * (float)s,
                                             -0.6f + 0.07f * (float)s,
                                              1.0f));
            float hspd  = fract1(native_sin((float)s * 3.77f + 1.23f) * 2345.6789f);
            float speed = SPIKE_DRIFT * (1.2f + 0.6f * hspd) / (0.4f + size);
            dir = rot_axis(base, axis, t * speed);
        }

        dir = escape_holes(dir, hole_dir, hole_ang);
        ctx->spike_c[i] = dir;

        float w_i = SPIKE_W * (0.7f + 0.4f * size);
        float cw  = native_cos(w_i);
        ctx->spike_cos_w[i]   = cw;
        ctx->spike_inv_ocw[i] = 1.0f / (1.0f - cw);

        float mask     = rough_mask(dir, t);
        float gate     = smoothstep(-0.05f, 0.25f, mask);
        float osc_rate = SPIKE_RATE / (0.5f + 0.5f * size);
        float phase    = t * osc_rate + (float)i * 1.37f;
        float osc      = 0.5f + 0.5f * native_sin(phase);
        ctx->spike_h[i] = SPIKE_H * size * gate * osc;
    }

    // Solo-spike relaxation --------------------------------------------------
    // After initial placement + hole-avoidance, solo spikes tend to ring
    // up along hole edges and leave wide membrane patches bare. Two
    // Gauss-Seidel iterations with soft tangential repulsion (from other
    // solos AND from cluster seeds) spread them more evenly without
    // disrupting clusters or pushing anything into a hole. Push magnitude
    // is kept small so the per-frame dynamics track the slow drift rather
    // than introducing visible jitter.
    //
    // cos(0.32 rad) = 0.94924 (solo minimum separation)
    // cos(0.25 rad) = 0.96891 (solo avoidance of cluster seeds)
    const float MIN_SOLO_SEP_COS     = 0.94924f;
    const float CLUSTER_AVOID_COS    = 0.96891f;
    const float SOLO_PUSH            = 0.40f;

    for (int iter = 0; iter < 2; iter++) {
        for (int i = 0; i < N_SOLO; i++) {
            float3 vi = ctx->spike_c[i];
            float3 force = (float3)(0.0f, 0.0f, 0.0f);

            // Repulsion from other solos
            for (int j = 0; j < N_SOLO; j++) {
                if (i == j) continue;
                float3 vj = ctx->spike_c[j];
                float cij = dot(vi, vj);
                if (cij > MIN_SOLO_SEP_COS) {
                    float3 tn = tangent_away(vi, vj);
                    force += tn * (cij - MIN_SOLO_SEP_COS);
                }
            }

            // Stronger repulsion from cluster seeds so solos stay clear
            for (int cc = 0; cc < N_CLUSTERS; cc++) {
                float cv = dot(vi, cluster_seeds[cc]);
                if (cv > CLUSTER_AVOID_COS) {
                    float3 tn = tangent_away(vi, cluster_seeds[cc]);
                    force += tn * (cv - CLUSTER_AVOID_COS) * 2.0f;
                }
            }

            float flen = length(force);
            if (flen > 1e-4f) {
                vi = normalize(vi + force * SOLO_PUSH);
                vi = escape_holes(vi, hole_dir, hole_ang);
                ctx->spike_c[i] = vi;
            }
        }
    }
}

// ============================================================================
// Spike contribution
// ============================================================================
// Sum of cone-profile protrusions at the discrete spike centers. Only
// added to the outer surface; inner stays smooth so veins can attach to
// clean membrane. Profile: `h * depth^SHARP` where `depth = 1` at the
// spike peak and `depth = 0` at the base angle SPIKE_W. Raising to the
// SHARP-th power makes the silhouette read as a pointy spike rather than
// a rounded bump.
static float spike_outer(float3 dir, __private const SceneCtx* ctx) {
    float s = 0.0f;
    for (int i = 0; i < N_SPIKES; i++) {
        float c = dot(dir, ctx->spike_c[i]);
        float cw = ctx->spike_cos_w[i];
        if (c > cw) {
            float depth = (c - cw) * ctx->spike_inv_ocw[i];
            s += ctx->spike_h[i] * native_powr(depth, SPIKE_SHARP);
        }
    }
    return s;
}

// ============================================================================
// Scene distance
// ============================================================================
// tag: 0 = shell, 1 = core, 2 = vein
static float map_scene(float3 p, __private const SceneCtx* ctx, int* tag_out) {
    float r = length(p);
    float3 d = (r > 1e-4f) ? (p / r) : (float3)(0.0f, 1.0f, 0.0f);

    // Shell -------------------------------------------------------------
    float i_disp = inner_disp(d, ctx->t);
    float o_disp = i_disp + spike_outer(d, ctx);

    // Thickness modulation: thinner near hole rims, baseline elsewhere.
    // For each hole, smoothstep over the last ~10 deg of arc outside its
    // angular footprint ramps `nearness` from 0 to 1 as we approach the
    // hole edge; take the max across all holes. 40% thinning at the edge.
    float hole_near = 0.0f;
    for (int i = 0; i < N_HOLES; i++) {
        float c = dot(d, ctx->hole_c[i]) * (1.0f / R_SHELL);
        float n = smoothstep(ctx->hole_cos_b[i] - 0.12f,
                              ctx->hole_cos_b[i], c);
        hole_near = fmax(hole_near, n);
    }
    float eff_thick = THICK * (1.0f - 0.40f * hole_near);

    float R_out = R_SHELL + eff_thick * 0.5f + o_disp;
    float R_in  = R_SHELL - eff_thick * 0.5f + i_disp;
    float d_outer = r - R_out;
    float d_inner = r - R_in;
    float d_shell = fmax(d_outer, -d_inner);

    // Hole subtraction --------------------------------------------------
    // Each hole's effective radius is perturbed by two sinusoidal layers
    // evaluated in the hole's local tangent plane (pk1, pk2). The layers
    // use different frequencies and unrelated phases, giving a 2D-varying
    // boundary pattern -- the hole reads as "roughly circular" but not
    // a perfect circle. Per-hole phases are fixed, so the shape stays
    // constant as the hole drifts.
    float d_hole = 1e10f;
    for (int i = 0; i < N_HOLES; i++) {
        float3 pl   = p - ctx->hole_c[i];
        float  plen = length(pl);
        float3 dir  = (plen > 1e-6f) ? (pl / plen) : (float3)(0.0f, 1.0f, 0.0f);
        float u = dot(dir, ctx->hole_pk1[i]);
        float v = dot(dir, ctx->hole_pk2[i]);
        float n = native_sin(u * 4.0f + ctx->hole_pp1[i]) * 0.55f
                + native_sin(v * 6.0f + ctx->hole_pp2[i]) * 0.35f;
        float r_eff = ctx->hole_r[i] * (1.0f + 0.18f * n);
        d_hole = fmin(d_hole, plen - r_eff);
    }
    float d_shell_carved = fmax(d_shell, -d_hole);

    // Core + veins + hubs + bridges --------------------------------------
    // Core radius pulses with the cardiac waveform; at peak systole the
    // surface swells by HEART_PUMP_AMP (~12%) which briefly engulfs the
    // small core-side hub spheres (smin blends them cleanly).
    float R_core_t = R_CORE * (1.0f + HEART_PUMP_AMP * heart_pump(ctx->t));
    float d_core   = r - R_core_t;

    // Accumulate the web-like interior structure as a single SDF blob.
    // Strategy: veins smin-unioned tightly, then hubs melted into it with
    // a bigger k so junctions read as swellings, then bridges joined with
    // a smaller k so they stay visually distinct secondary fibers.
    float d_veins = 1e10f;
    for (int i = 0; i < N_VEINS; i++) {
        float dv = sd_curved_vein(p,
                                   ctx->vein_a[i],
                                   ctx->vein_m1[i],
                                   ctx->vein_m2[i],
                                   ctx->vein_m3[i],
                                   ctx->vein_b[i],
                                   VEIN_R0, VEIN_R1,
                                   ctx->vein_seed[i],
                                   ctx->t);
        d_veins = smin_poly(d_veins, dv, 0.020f);
    }
    // Only CORE hubs are rendered as sphere primitives -- they read as
    // nodes where veins converge at the core and the user likes them there.
    // Membrane hubs serve only as shared endpoint anchors for veins; they
    // don't contribute their own SDF geometry, so there's no sphere to bulge
    // out of the membrane. Veins just taper into the surface and the final
    // smin handles the smooth fuse.
    for (int i = 0; i < N_HUBS_C; i++) {
        float dh = length(p - ctx->hub_p[i]) - ctx->hub_r[i];
        d_veins = smin_poly(d_veins, dh, 0.025f);
    }
    for (int i = 0; i < N_BRIDGES; i++) {
        float db = sd_capsule(p, ctx->bridge_a[i], ctx->bridge_b[i], BRIDGE_R);
        d_veins = smin_poly(d_veins, db, 0.015f);
    }

    float d_inside   = smin_poly(d_core, d_veins, 0.04f);
    int   inside_tag = (d_core < d_veins) ? 1 : 2;

    // Final pick --------------------------------------------------------
    // Smooth-union the membrane shell with the interior structure: where
    // hubs/veins touch the inner surface, the smin builds a soft flare so
    // the junction reads as "opening up and fusing" rather than two
    // surfaces meeting at a hard line. FUSE_K is small (~3 cm at R=1) so
    // the blend only affects points close to the actual contact region.
    float d_final = smin_poly(d_shell_carved, d_inside, FUSE_K);
    if (tag_out) *tag_out = (d_shell_carved < d_inside) ? 0 : inside_tag;
    return d_final;
}

static float map(float3 p, __private const SceneCtx* ctx) {
    int dummy;
    return map_scene(p, ctx, &dummy);
}

static float3 sdf_normal(float3 p, __private const SceneCtx* ctx) {
    const float h = 0.0025f;
    const float3 k0 = (float3)( 1.0f, -1.0f, -1.0f);
    const float3 k1 = (float3)(-1.0f, -1.0f,  1.0f);
    const float3 k2 = (float3)(-1.0f,  1.0f, -1.0f);
    const float3 k3 = (float3)( 1.0f,  1.0f,  1.0f);
    return normalize(k0 * map(p + k0 * h, ctx)
                   + k1 * map(p + k1 * h, ctx)
                   + k2 * map(p + k2 * h, ctx)
                   + k3 * map(p + k3 * h, ctx));
}

// ============================================================================
// Ray-sphere bounding
// ============================================================================
static float2 ray_sphere(float3 ro, float3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return (float2)(1.0f, -1.0f);
    float s = native_sqrt(disc);
    return (float2)(-b - s, -b + s);
}

// ============================================================================
// KERNEL
// ============================================================================
__kernel void raymarch_anomaly(
    __global float4* pixels,
    const int width, const int height,
    const float3 eye,
    const float3 fwd,
    const float3 right,
    const float3 up,
    const float fov_tan,
    const float sim_time,
    const float amp,
    const int diag_mode
) {
    int idx = get_global_id(0);
    if (idx >= width * height) return;
    (void)amp;

    // Build the scene context once per thread. All pixels in a frame share
    // the same (t-only) inputs so this is constant work -- we pay N_VEINS
    // FBM evaluations + the hole-avoidance loop once instead of per-step.
    SceneCtx ctx;
    build_ctx(&ctx, sim_time);

    int px = idx % width;
    int py = idx / width;

    float aspect = (float)width / (float)height;
    float ndc_x =  (2.0f * ((float)px + 0.5f) / (float)width  - 1.0f);
    float ndc_y = -(2.0f * ((float)py + 0.5f) / (float)height - 1.0f);

    float3 rd = normalize(fwd
                        + right * (ndc_x * fov_tan * aspect)
                        + up    * (ndc_y * fov_tan));
    float3 ro = eye;

    // Max spike size factor is ~3.0 (0.5 + 2.5*1.0); pad for safety.
    float R_outer = R_SHELL + THICK * 0.5f + DISP_AMP + SPIKE_AMP + SPIKE_H * 3.1f + 0.05f;
    float2 ts = ray_sphere(ro, rd, R_outer);
    if (ts.x > ts.y) {
        pixels[idx] = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }
    float t_near = fmax(ts.x, 0.0f);
    float t_far  = ts.y;

    float t = t_near;
    float hit_t   = -1.0f;
    float3 hit_p  = (float3)(0.0f);
    int   hit_tag = -1;
    const float HIT_EPS = 0.0012f;

    for (int i = 0; i < MAX_STEPS; i++) {
        float3 p = ro + rd * t;
        int tag;
        float d = map_scene(p, &ctx, &tag);
        if (d < HIT_EPS) { hit_t = t; hit_p = p; hit_tag = tag; break; }
        t += fmax(d * 0.50f, 0.0020f);  // sharp spike bases violate Lipschitz-1 heavily
        if (t > t_far) break;
    }

    float3 col = (float3)(0.0f);

    if (hit_t > 0.0f) {
        float3 n = sdf_normal(hit_p, &ctx);

        if (diag_mode == 1) {
            col = n * 0.5f + 0.5f;
        } else if (diag_mode == 2) {
            if      (hit_tag == 0) col = (float3)(0.85f, 0.25f, 0.25f); // shell
            else if (hit_tag == 1) col = (float3)(0.25f, 0.85f, 0.30f); // core
            else                   col = (float3)(0.30f, 0.40f, 0.95f); // vein
        } else {
            // Half-Lambert squared + low ambient: back never fully black,
            // geometry readable from any angle.
            float3 L = normalize((float3)(0.4f, 0.8f, 0.4f));
            float dl    = 0.5f + 0.5f * dot(n, L);
            float shade = dl * dl;
            col = (float3)(0.12f + 0.88f * shade);
        }
    }

    col.x = native_powr(fmax(col.x, 0.0f), 1.0f / 2.2f);
    col.y = native_powr(fmax(col.y, 0.0f), 1.0f / 2.2f);
    col.z = native_powr(fmax(col.z, 0.0f), 1.0f / 2.2f);

    pixels[idx] = (float4)(col, 1.0f);
}
