
// struct Track {
//     unsigned int hitsNum;
//     unsigned int hits[MAX_TRACK_SIZE];
// };

// struct TrackParameters {
//     float x0;
//     float y0;
//     float tx;
//     float ty;
// };


// Fit
void fit(__global const float* const hit_Xs,
  __global const float* const hit_Ys,
  __global const float* const hit_Zs,
  __global struct Track* const tracks,
  const int trackno,
  __global struct TrackParameters* const trackParams)
{
  float s0, sx, sz, sxz, sz2;
  float u0, uy, uz, uyz, uz2;
  s0 = sx = sz = sxz = sz2 = 0.0f;
  u0 = uy = uz = uyz = uz2 = 0.0f;
  
  // Iterate over the hits, and fit
  struct Track t = tracks[trackno];

  for (int h=0; h<t.hitsNum; ++h) {
    const float x = hit_Xs[h];
    const float y = hit_Ys[h];
    const float z = hit_Zs[h];
    
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
  }

  float dens = 1.0f / (sz2 * s0 - sz * sz);
  if (fabs(dens) > 10e+10) dens = 1.0f;
  trackParams[trackno].tx = (sxz * s0 - sx * sz) * dens;
  trackParams[trackno].x0 = (sx * sz2 - sxz * sz) * dens;

  float denu = 1.0f / (uz2 * u0 - uz * uz);
  if (fabs(denu) > 10e+10) denu = 1.0f;
  trackParams[trackno].ty = (uyz * u0 - uy * uz) * denu;
  trackParams[trackno].y0 = (uy * uz2 - uyz * uz) * denu;
}

//=========================================================================
// Return the covariance matrix of the last fit at the specified z
//=========================================================================
void covariance(__global float* covariance, const double z) const {
  // Gaudi::TrackSymMatrix cov;
  // //== Ad hoc matrix inversion, as it is almost diagonal!
  // const double m00 = s0;
  // const double m11 = u0;
  // const double m20 = sz - z * s0;
  // const double m31 = uz - z * u0;
  // const double m22 = sz2 - 2 * z * sz + z * z * s0;
  // const double m33 = uz2 - 2 * z * uz + z * z * u0;
  // const double den20 = 1.0 / (m22 * m00 - m20 * m20);
  // const double den31 = 1.0 / (m33 * m11 - m31 * m31);

  // cov(0, 0) = m22 * den20;
  // cov(2, 0) = -m20 * den20;
  // cov(2, 2) = m00 * den20;

  // cov(1, 1) = m33 * den31;
  // cov(3, 1) = -m31 * den31;
  // cov(3, 3) = m11 * den31;

  // cov(4, 4) = 1.;
  // return cov;
}

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
  for (int i=0; i<(number_of_tracks + id_x - 1) / id_x; ++i) {
    const int element = id_x + i * blockDim;
    if (element < number_of_tracks) {
        fit(hit_Xs, hit_Ys, hit_Zs, tracks, element, track_parameters);
    }
  }
}
