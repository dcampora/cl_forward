
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

struct Track {
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

struct FitKalmanTrackParameters {
  float x, y, z, tx, ty;
  struct Covariance cov;
  float chi2;
};

/**
 * Helper function to filter one hit
 * @param  z       [description]
 * @param  x       [description]
 * @param  tx      [description]
 * @param  covXX   [description]
 * @param  covXTx  [description]
 * @param  covTxTx [description]
 * @param  zhit    [description]
 * @param  xhit    [description]
 * @param  whit    [description]
 * @return         [description]
 */
float filter(
  float z, float* x, float* tx,
  float* covXX, float* covXTx, float* covTxTx,
  float zhit, float xhit, float whit)
{
  // compute the prediction
  const float dz = zhit - z;
  const float predx = (*x) + dz * (*tx);

  const float dz_t_covTxTx = dz * (*covTxTx);
  const float predcovXTx = (*covXTx) + dz_t_covTxTx;
  const float dx_t_covXTx = dz * (*covXTx);

  const float predcovXX = (*covXX) + 2 * dx_t_covXTx + dz * dz_t_covTxTx;
  const float predcovTxTx = (*covTxTx);
  // compute the gain matrix
  const float R = 1.0f / ((1.0f / whit) + predcovXX);
  const float Kx = predcovXX * R;
  const float KTx = predcovXTx * R;
  // update the state vector
  const float r = xhit - predx;
  x[0] = predx + Kx * r;
  tx[0] = (*tx) + KTx * r;
  // update the covariance matrix. we can write it in many ways ...
  covXX[0] /*= predcovXX  - Kx * predcovXX */ = (1 - Kx) * predcovXX;
  covXTx[0] /*= predcovXTx - predcovXX * predcovXTx / R */ = (1 - Kx) * predcovXTx;
  covTxTx[0] = predcovTxTx - KTx * predcovXTx;
  // return the chi2
  return r * r * R;
}

//===============================================================================
// Fit the track with a Kalman filter, allowing for some scattering at
// every hit. Function arguments:
//  state: state at the last filtered hit.
//  direction=+1 : filter in positive z direction (not normally what you want)
//  direction=-1 : filter in negative z direction
//  noise2PerLayer: scattering contribution (squared) to tx and ty
// The return value is the chi2 of the fit
// ===============================================================================
void fitKalman(__global const float* const hit_Xs,
  __global const float* const hit_Ys,
  __global const float* const hit_Zs,
  __global struct Track* const tracks,
  const int trackno,
  __global struct TrackParameters* const track_parameters,
  __global struct FitKalmanTrackParameters* const fit_kalman_track_parameters,
  const int is_upstream)
{
  // We do assume it is ordered by Z, as it is
  struct Track t = tracks[trackno];
  struct TrackParameters tp = track_parameters[trackno];
  struct FitKalmanTrackParameters fktp;

  // Parameters are calculated here
  const int direction = (tp.backward ? 1 : -1) * (is_upstream==1 ? 1 : -1);
  const float noise2PerLayer = 1e-8 + 7e-6 * (tp.tx * tp.tx + tp.ty * tp.ty);

  // assume the hits are sorted,
  // but don't assume anything on the direction of sorting
  int firsthit = 0;
  int lasthit = t.hitsNum - 1;
  int dhit = 1;
  if ((hit_Zs[t.hits[lasthit]] - hit_Zs[t.hits[firsthit]]) * direction < 0) {
    const int temp = firsthit;
    firsthit = lasthit;
    lasthit = temp;
    dhit = -1;
  }

  // We filter x and y simultaneously but take them uncorrelated.
  // filter first the first hit.

  // const PrPixelHit *hit = m_hits[firsthit];
  const int hitno = t.hits[firsthit];
  float x = hit_Xs[hitno];
  float y = hit_Ys[hitno];
  float z = hit_Zs[hitno];
  float tx = tp.tx;
  float ty = tp.ty;

  // initialize the covariance matrix
  // float covXX = 1 / hit->wx();
  // float covYY = 1 / hit->wy();
  float covXX = PARAM_W_INVERTED;
  float covYY = PARAM_W_INVERTED;
  float covXTx = 0.0f;  // no initial correlation
  float covYTy = 0.0f;
  float covTxTx = 1.0f;  // randomly large error
  float covTyTy = 1.0f;

  // add remaining hits
  float chi2 = 0.0f;
  for (int i=firsthit + dhit; i!=lasthit + dhit; i+=dhit) {
    const int hitno = t.hits[i];
    const float hit_x = hit_Xs[hitno];
    const float hit_y = hit_Ys[hitno];
    const float hit_z = hit_Zs[hitno];
    
    // add the noise
    covTxTx += noise2PerLayer;
    covTyTy += noise2PerLayer;

    // filter X and filter Y
    chi2 += filter(z, &x, &tx, &covXX, &covXTx, &covTxTx, hit_z, hit_x, PARAM_W);
    chi2 += filter(z, &y, &ty, &covYY, &covYTy, &covTyTy, hit_z, hit_y, PARAM_W);
    
    // update z (note done in the filter, since needed only once)
    z = hit_z;
  }

  // add the noise at the last hit
  covTxTx += noise2PerLayer;
  covTyTy += noise2PerLayer;

  // finally, fill the state
  fktp.x = x;
  fktp.y = y;
  fktp.z = z;
  fktp.tx = tx;
  fktp.ty = ty;
  fktp.cov.c00 = covXX;
  fktp.cov.c20 = covXTx;
  fktp.cov.c22 = covTxTx;
  fktp.cov.c11 = covYY;
  fktp.cov.c31 = covYTy;
  fktp.cov.c33 = covTyTy;
  fktp.chi2 = chi2;

  fit_kalman_track_parameters[trackno] = fktp;
}

/**
 * fitTracks
 * @param dev_input            [description]
 * @param dev_event_offsets    [description]
 * @param dev_tracks           [description]
 * @param dev_track_parameters [description]
 * @param dev_atomicsStorage   [description]
 */
__kernel void fitKalmanTracks(
  __global const char* const dev_input,
  __global int* const dev_event_offsets,
  __global struct Track* const dev_tracks,
  __global struct TrackParameters* const dev_track_parameters,
  __global int* const dev_atomicsStorage,
  __global struct FitKalmanTrackParameters* const dev_fit_kalman_track_parameters,
  const int is_upstream)
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
  __global struct FitKalmanTrackParameters* fit_kalman_track_parameters = dev_fit_kalman_track_parameters + tracks_offset;

  // We will process n tracks with m threads (workers)
  const int number_of_tracks = dev_atomicsStorage[event_number];
  const int id_x = get_local_id(0);
  const int blockDim = get_local_size(0);

  // Calculate track no, and iterate over the tracks
  for (int i=0; i<(number_of_tracks + blockDim - 1) / blockDim; ++i) {
    const int element = id_x + i * blockDim;
    if (element < number_of_tracks) {
        fitKalman(hit_Xs, hit_Ys, hit_Zs, tracks, element,
          track_parameters, fit_kalman_track_parameters, is_upstream);
    }
  }
}
