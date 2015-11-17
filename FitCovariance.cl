
// Used to prefer a device type over another one
#define DEVICE_CPU 0
#define DEVICE_GPU 1
#define DEVICE_ACCELERATOR 2
#define DEVICE_PREFERENCE DEVICE_GPU
#define DEVICE_NUMBER 0

#define NUMTHREADS_X 64
#define MAX_NUMTHREADS_Y 16
#define NUM_ATOMICS 5
#define USE_SHARED_FOR_HITS false
#define SH_HIT_MULT 2

#define MAX_TRACKS 3000
#define MAX_TRACK_SIZE 24

#define REQUIRED_UNIQUES 0.6f
#define MIN_HITS_TRACK 3
#define MAX_FLOAT FLT_MAX
#define MIN_FLOAT -FLT_MAX
#define MAX_SKIPPED_MODULES 3
#define TTF_MODULO 2000

#define PARAM_W 3966.94f // 0.050 / sqrt( 12. )
#define PARAM_W_INVERTED 0.000252083f
#define PARAM_MAXXSLOPE 0.4f
#define PARAM_MAXYSLOPE 0.3f
#define PARAM_MAXXSLOPE_CANDIDATES 0.4f

#define PARAM_TOLERANCE 0.6f
#define PARAM_TOLERANCE_CANDIDATES 0.6f

#define MAX_SCATTER 0.000016f
#define SENSOR_DATA_HITNUMS 3
#define RESULTS_FOLDER "results"

#define PRINT_SOLUTION true
#define PRINT_VERBOSE true
#define ASSERTS_ENABLED false

#if ASSERTS_ENABLED == true
#include "assert.h"
#define ASSERT(EXPR) ASSERT_CL_RETURN(EXPR, #EXPR);
#else
#define ASSERT(EXPR) 
#endif

struct Sensor {
    unsigned int hitStart;
    unsigned int hitNums;
};

struct Hit {
    float x;
    float y;
    float z;
};

struct Track { // 4 + 24 * 4 = 100 B
    unsigned int hitsNum;
    unsigned int hits[MAX_TRACK_SIZE];
};

struct Covariance {
  float c00, c20, c22, c11, c31, c33;
};

struct TrackParameters {
  float x0, y0, tx, ty, chi2;
  struct Covariance cov;
  float zbeam;
  bool backward;
};

/**
 * Compute the track parameters and the covariance
 * @param hit_Xs      [description]
 * @param hit_Ys      [description]
 * @param hit_Zs      [description]
 * @param tracks      [description]
 * @param trackno     [description]
 * @param track_parameters [description]
 */
void fit(__global const float* const hit_Xs,
  __global const float* const hit_Ys,
  __global const float* const hit_Zs,
  __global struct Track* const tracks,
  const int trackno,
  __global struct TrackParameters* const track_parameters)
{
  //=========================================================================
  // Compute the track parameters
  //=========================================================================
  struct TrackParameters tp;
  float s0, sx, sz, sxz, sz2;
  float u0, uy, uz, uyz, uz2;
  float sum_xxw = 0.0f, sum_yyw = 0.0f;
  s0 = sx = sz = sxz = sz2 = 0.0f;
  u0 = uy = uz = uyz = uz2 = 0.0f;
  
  // Iterate over the hits, and fit
  struct Track t = tracks[trackno];

  for (int h=0; h<t.hitsNum; ++h) {
    const int hitno = t.hits[h];
    const float x = hit_Xs[hitno];
    const float y = hit_Ys[hitno];
    const float z = hit_Zs[hitno];
    
    const float wx = PARAM_W;
    const float wx_t_x = wx * x;
    const float wx_t_z = wx * z;
    s0 += wx;
    sx += wx_t_x;
    sz += wx_t_z;
    sxz += wx_t_x * z;
    sz2 += wx_t_z * z;

    const float wy = PARAM_W;
    const float wy_t_y = wy * y;
    const float wy_t_z = wy * z;
    u0 += wy;
    uy += wy_t_y;
    uz += wy_t_z;
    uyz += wy_t_y * z;
    uz2 += wy_t_z * z;

    sum_xxw += x*x*PARAM_W;
    sum_yyw += y*y*PARAM_W;
  }

  {
    float dens = 1.0f / (sz2 * s0 - sz * sz);
    if (fabs(dens) > 10e+10) dens = 1.0f;
    tp.tx = (sxz * s0 - sx * sz) * dens;
    tp.x0 = (sx * sz2 - sxz * sz) * dens;

    float denu = 1.0f / (uz2 * u0 - uz * uz);
    if (fabs(denu) > 10e+10) denu = 1.0f;
    tp.ty = (uyz * u0 - uy * uz) * denu;
    tp.y0 = (uy * uz2 - uyz * uz) * denu;
  }

  {
    //=========================================================================
    // Return the covariance matrix of the last fit at the specified z
    //=========================================================================
    tp.zbeam = -(tp.x0 * tp.tx + tp.y0 * tp.ty) / (tp.tx * tp.tx + tp.ty * tp.ty);

    const float m00 = s0;
    const float m11 = u0;
    const float m20 = sz - tp.zbeam * s0;
    const float m31 = uz - tp.zbeam * u0;
    const float m22 = sz2 - 2 * tp.zbeam * sz + tp.zbeam * tp.zbeam * s0;
    const float m33 = uz2 - 2 * tp.zbeam * uz + tp.zbeam * tp.zbeam * u0;
    const float den20 = 1.0f / (m22 * m00 - m20 * m20);
    const float den31 = 1.0f / (m33 * m11 - m31 * m31);

    tp.cov.c00 = m22 * den20;
    tp.cov.c20 = -m20 * den20;
    tp.cov.c22 = m00 * den20;
    tp.cov.c11 = m33 * den31;
    tp.cov.c31 = -m31 * den31;
    tp.cov.c33 = m11 * den31;
  }

  {
    // Define backward as z closest to beam downstream of hits.
    tp.backward = tp.zbeam > hit_Zs[t.hits[0]];
  }

  {
    //=========================================================================
    // Chi2 / degrees-of-freedom of straight-line fit
    //=========================================================================
    float ch = 0.0f;
    int nDoF = -4 + 2*t.hitsNum;
    for (int h=0; h<t.hitsNum; ++h) {
      const int hitno = t.hits[h];

      const float z = hit_Zs[hitno];
      const float x = tp.x0 + tp.tx * z;
      const float y = tp.y0 + tp.ty * z;

      // const float dx = x - hit_Xs[hitno];
      // const float dy = y - hit_Ys[hitno];
      // ch += dx * dx * PARAM_W + dy * dy * PARAM_W;

      // Error is not necessarily equal for all values in production
      ch += x*x*PARAM_W + y*y*PARAM_W;
      // nDoF += 2;
    }
    ch -= (sum_xxw + sum_yyw);
    tp.chi2 = ch / nDoF; 
  }

  tp.x0 = tp.x0 + tp.tx * tp.zbeam;
  tp.y0 = tp.y0 + tp.ty * tp.zbeam;

  track_parameters[trackno] = tp;
}

/**
 * fitTracks
 * @param dev_input            [description]
 * @param dev_event_offsets    [description]
 * @param dev_tracks           [description]
 * @param dev_track_parameters [description]
 * @param dev_atomicsStorage   [description]
 */
__kernel void fitTracks(
  __global const char* const dev_input,
  __global int* const dev_event_offsets,
  __global struct Track* const dev_tracks,
  __global struct TrackParameters* const dev_track_parameters,
  __global int* const dev_atomicsStorage)
{
  // Data initialization
  // Each event is treated with two blocks, one for each side.
  const int event_number = get_group_id(0);
  const int events_under_process = get_num_groups(0);
  const int tracks_offset = event_number * MAX_TRACKS;

  // Pointers to data within the event
  const int data_offset = dev_event_offsets[event_number];
  __global const int* const no_sensors = (__global const int*) (dev_input + data_offset);
  __global const int* const no_hits = (__global const int*) (no_sensors + 1);
  __global const int* const sensor_Zs = (__global const int*) (no_hits + 1);
  const int number_of_sensors = no_sensors[0];
  const int number_of_hits = no_hits[0];
  __global const int* const sensor_hitStarts = (__global const int*) (sensor_Zs + number_of_sensors);
  __global const int* const sensor_hitNums = (__global const int*) (sensor_hitStarts + number_of_sensors);
  __global const unsigned int* const hit_IDs = (__global const unsigned int*) (sensor_hitNums + number_of_sensors);
  __global const float* const hit_Xs = (__global const float*) (hit_IDs + number_of_hits);
  __global const float* const hit_Ys = (__global const float*) (hit_Xs + number_of_hits);
  __global const float* const hit_Zs = (__global const float*) (hit_Ys + number_of_hits);

  // Per event datatypes
  __global struct Track* tracks = dev_tracks + tracks_offset;
  __global struct TrackParameters* track_parameters = dev_track_parameters + tracks_offset;

  // We will process n tracks with m threads (workers)
  const int number_of_tracks = dev_atomicsStorage[event_number];
  const int id_x = get_local_id(0);
  const int blockDim = get_local_size(0);

  // Calculate track no, and iterate over the tracks
  for (int i=0; i<(number_of_tracks + blockDim - 1) / blockDim; ++i) {
    const int element = id_x + i * blockDim;
    if (element < number_of_tracks) {
        fit(hit_Xs, hit_Ys, hit_Zs, tracks, element, track_parameters);
    }
  }
}
