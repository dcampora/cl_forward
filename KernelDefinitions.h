#ifndef OPENCL_OPTIONS
#define OPENCL_OPTIONS 1

#define DEVICE_NUMBER 0
#define DEVICE_CPU 0
#define DEVICE_GPU 1
#define DEVICE_ACCELERATOR 2
#define DEVICE_PREFERENCE DEVICE_CPU

#define NUMTHREADS_X 8
#define TF_NUMTHREADS_X 8
#define TF_NUMTHREADS_Y 1
#define NUM_ATOMICS 5
#define USE_SHARED_FOR_HITS false
#define SH_HIT_MULT 2

#define MAX_TRACKS 6000
#define MAX_TRACK_SIZE 24
#define MAX_TRACKS_PER_SENSOR 300

#define MIN_HITS_TRACK 3
#define MAX_FLOAT FLT_MAX
#define MIN_FLOAT -FLT_MAX
#define MAX_SKIPPED_MODULES 3
#define TTF_MODULO 2000

#define PARAM_W 3966.94f // (std::sqrt(12.0) / (0.055)) ^ 2
#define PARAM_W_INVERTED 0.000252083f
#define PARAM_MAXXSLOPE 0.7f
#define PARAM_MAXYSLOPE 0.7f
#define PARAM_MAXXSLOPE_CANDIDATES 0.7f

#define PARAM_TOLERANCE 0.6f
#define PARAM_TOLERANCE_CANDIDATES 0.6f

#define MAX_SCATTER 0.000016f
#define SENSOR_DATA_HITNUMS 3

#define PRINT_SOLUTION true
#define PRINT_VERBOSE true
#define RESULTS_FOLDER "results"

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
  float x0, y0, tx, ty;
  struct Covariance cov;
  float zbeam;
  float chi2;
  bool backward;
};

struct FitKalmanTrackParameters {
  float x, y, z, tx, ty;
  struct Covariance cov;
  float chi2;
};

#define CL_Sensor Sensor
#define CL_Hit Hit
#define CL_Track Track
#define CL_Covariance Covariance
#define CL_TrackParameters TrackParameters
#define CL_FitKalmanTrackParameters FitKalmanTrackParameters

#endif
