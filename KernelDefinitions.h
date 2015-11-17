
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

struct FitKalmanTrackParameters {
  float x, y, z, tx, ty;
  struct Covariance cov;
  float chi2;
};
