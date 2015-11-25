
__kernel void clCloneKiller(__global struct Track* const dev_tracks, __global const char* const dev_input,
  __global int* const dev_tracks_to_follow, __global bool* const dev_hit_used,
  __global int* const dev_atomicsStorage, __global struct Track* const dev_tracklets,
  __global int* const dev_weak_tracks, __global int* const dev_event_offsets,
  __global int* const dev_hit_offsets, __global float* const dev_best_fits,
  __global int* const dev_hit_candidates, __global int* const dev_hit_h2_candidates,
  __global struct Track* const dev_tracks_per_sensor) {
  
  // Data initialization
  // Each event is treated with two blocks, one for each side.
  const int sensor_id = get_group_id(0);
  const int sensors_under_process = get_num_groups(0);
  const int blockDim_product = get_local_size(0) * get_local_size(1);

  // Pointers to data within the event
  __global const int* const no_sensors = (__global const int*) dev_input;
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

  // Per side datatypes
  __global bool* const hit_used = dev_hit_used;
  __global int* const hit_candidates = dev_hit_candidates;
  __global int* const hit_h2_candidates = dev_hit_h2_candidates;

  __global int* const tracks_to_follow = dev_tracks_to_follow + sensor_id * TTF_MODULO;
  __global struct Track* const tracklets = dev_tracklets;
  __global float* const best_fits = dev_best_fits + sensor_id * blockDim_product;

  __global unsigned int* const tracks_insertPointer = (__global unsigned int*) dev_atomicsStorage;
  __global unsigned int* const weaktracks_insertPointer = (__global unsigned int*) dev_atomicsStorage + 1;
  __global int* const weak_tracks = dev_weak_tracks;

  __global struct Track* const tracks_per_sensor = dev_tracks_per_sensor;
  __global struct Track* const tracks = dev_tracks;

  // Initialize variables according to event number and sensor side
  // Insert pointers (atomics)
  int shift = 2;
  __global int* const ttf_insertPointer = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const tracklets_insertPointer = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const sh_hit_lastPointer = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const max_numhits_to_process = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const tracks_per_sensor_insertPointer = (__global int*) dev_atomicsStorage + shift + sensor_id;

  // The fun begins
#if USE_SHARED_FOR_HITS
  __local float sh_hit_x [NUMTHREADS_X * SH_HIT_MULT];
  __local float sh_hit_y [NUMTHREADS_X * SH_HIT_MULT];
  __local float sh_hit_z [NUMTHREADS_X * SH_HIT_MULT];
#endif
  __local int sensor_data [6];

  // Process all tracks
  for (int i=0; i<number_of_sensors-4; ++i) {
    int first_sensor = 51-i;

    const int number_of_tracks = tracks_per_sensor_insertPointer[i];
    for (int j=0; j<(number_of_tracks + blockDim_product - 1) / blockDim_product; ++j) {
      const unsigned int trackno = blockDim_product * j + get_local_id(1) * get_local_size(0) + get_local_id(0);
      struct Track t;
      bool used = true;

      if (trackno < number_of_tracks) {
        t = (tracks_per_sensor + i*MAX_TRACKS_PER_SENSOR)[trackno];
        used = hit_used[t.hits[0]] || hit_used[t.hits[1]];
      }

      barrier(CLK_GLOBAL_MEM_FENCE);

      if (!used) {
        const unsigned int insp = atomic_add(tracks_insertPointer, 1);
        tracks[insp] = t;

        for (int k=0; k<t.hitsNum; ++k) {
          hit_used[t.hits[k]] = 1;
        }
      }
    }
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  // Process weak tracks
  // Compute the three-hit tracks left
  const unsigned int weaktracks_total = weaktracks_insertPointer[0];
  for (int i=0; i<(weaktracks_total + blockDim_product - 1) / blockDim_product; ++i) {
    const unsigned int weaktrack_no = blockDim_product * i + get_local_id(1) * get_local_size(0) + get_local_id(0);
    if (weaktrack_no < weaktracks_total) {

      // Load the tracks from the tracklets
      const int fulltrackno = weak_tracks[weaktrack_no];
      const int sensor_id_wt = fulltrackno >> 24;
      const int trackletno = fulltrackno & 0x00FFFFFF;
      const struct Track t = (tracklets + sensor_id_wt*MAX_TRACKS_PER_SENSOR)[trackletno];

      // Store them in the tracks bag iff they
      // are made out of three unused hits
      if (!hit_used[t.hits[0]] &&
          !hit_used[t.hits[1]] &&
          !hit_used[t.hits[2]]) {
        const unsigned int trackno = atomic_add(tracks_insertPointer, 1);
        tracks[trackno] = t;
      }
    }
  }
}
