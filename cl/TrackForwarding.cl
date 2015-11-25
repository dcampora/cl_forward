
float fitHitToTrack(const float tx, const float ty, const struct Hit* h0, const float h1_z, const struct Hit* h2) {
  // tolerances
  const float dz = h2->z - h0->z;
  const float x_prediction = h0->x + tx * dz;
  const float dx = fabs(x_prediction - h2->x);
  const bool tolx_condition = dx < PARAM_TOLERANCE;

  const float y_prediction = h0->y + ty * dz;
  const float dy = fabs(y_prediction - h2->y);
  const bool toly_condition = dy < PARAM_TOLERANCE;

  // Scatter - Updated to last PrPixel
  const float scatterNum = (dx * dx) + (dy * dy);
  const float scatterDenom = 1.f / (h2->z - h1_z);
  const float scatter = scatterNum * scatterDenom * scatterDenom;

  const bool scatter_condition = scatter < MAX_SCATTER;
  const bool condition = tolx_condition && toly_condition && scatter_condition;

  return condition * scatter + !condition * MAX_FLOAT;
}

void trackForwarding(
#if USE_SHARED_FOR_HITS
  __local float* const sh_hit_x,
  __local float* const sh_hit_y,
  __local float* const sh_hit_z,
#endif
  __global const float* const hit_Xs,
  __global const float* const hit_Ys,
  __global const float* const hit_Zs,
  __global bool* const hit_used,
  volatile __global int* const tracks_insertPointer,
  volatile __global int* const ttf_insertPointer,
  __global int* const weaktracks_insertPointer,
  const int blockDim_sh_hit,
  __local int* const sensor_data,
  const unsigned int diff_ttf,
  const int blockDim_product,
  __global int* const tracks_to_follow,
  __global int* const weak_tracks,
  const unsigned int prev_ttf,
  __global struct Track* const tracklets,
  __global struct Track* const tracks,
  const int sensor_id) {

  for (int i=0; i<(diff_ttf + blockDim_product - 1) / blockDim_product; ++i) {
    const unsigned int ttf_element = blockDim_product * i + get_local_id(1) * get_local_size(0) + get_local_id(0);

    // These variables need to go here, shared memory and scope requirements
    float tx, ty, h1_z;
    unsigned int trackno, fulltrackno, skipped_modules, best_hit_h2;
    struct Track t;
    struct Hit h0;

    // The logic is broken in two parts for shared memory loading
    const bool ttf_condition = ttf_element < diff_ttf;
    if (ttf_condition) {
      fulltrackno = tracks_to_follow[(prev_ttf + ttf_element) % TTF_MODULO];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      skipped_modules = (fulltrackno & 0x70000000) >> 28;
      trackno = fulltrackno & 0x0FFFFFFF;

      __global const struct Track* const track_pointer = track_flag ? tracklets : tracks;

      t = track_pointer[trackno];

      // Load last two hits in h0, h1
      const int t_hitsNum = t.hitsNum;
      const int h0_num = t.hits[t_hitsNum - 2];
      const int h1_num = t.hits[t_hitsNum - 1];

      h0.x = hit_Xs[h0_num];
      h0.y = hit_Ys[h0_num];
      h0.z = hit_Zs[h0_num];

      const float h1_x = hit_Xs[h1_num];
      const float h1_y = hit_Ys[h1_num];
      h1_z = hit_Zs[h1_num];

      // Track forwarding over t, for all hits in the next module
      // Line calculations
      const float td = 1.0f / (h1_z - h0.z);
      const float txn = (h1_x - h0.x);
      const float tyn = (h1_y - h0.y);
      tx = txn * td;
      ty = tyn * td;
    }

    // Search for a best fit
    // Load shared elements
    
    // Iterate in the third list of hits
    // Tiled memory access on h2
    // Only load for get_local_id(1) == 0
    float best_fit = MAX_FLOAT;
    for (int k=0; k<(sensor_data[SENSOR_DATA_HITNUMS + 2] + blockDim_sh_hit - 1) / blockDim_sh_hit; ++k) {
      
#if USE_SHARED_FOR_HITS
      barrier(CLK_LOCAL_MEM_FENCE);
      const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
      const int sh_hit_no = blockDim_sh_hit * k + tid;
      if (get_local_id(1) < SH_HIT_MULT && sh_hit_no < sensor_data[SENSOR_DATA_HITNUMS + 2]) {
        const int h2_index = sensor_data[2] + sh_hit_no;

        // Coalesced memory accesses
        sh_hit_x[tid] = hit_Xs[h2_index];
        sh_hit_y[tid] = hit_Ys[h2_index];
        sh_hit_z[tid] = hit_Zs[h2_index];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

      if (ttf_condition) {
        const int last_hit_h2 = min(blockDim_sh_hit * (k + 1), sensor_data[SENSOR_DATA_HITNUMS + 2]);
        for (int kk=blockDim_sh_hit * k; kk<last_hit_h2; ++kk) {
          
          const int h2_index = sensor_data[2] + kk;
#if USE_SHARED_FOR_HITS
          const int sh_h2_index = kk % blockDim_sh_hit;
          struct Hit h2;
          h2.x = sh_hit_x[sh_h2_index];
          h2.y = sh_hit_y[sh_h2_index];
          h2.z = sh_hit_z[sh_h2_index];
#else
          struct Hit h2;
          h2.x = hit_Xs[h2_index];
          h2.y = hit_Ys[h2_index];
          h2.z = hit_Zs[h2_index];
#endif

          const float fit = fitHitToTrack(tx, ty, &h0, h1_z, &h2);
          const bool fit_is_better = fit < best_fit;

          best_fit = fit_is_better * fit + !fit_is_better * best_fit;
          best_hit_h2 = fit_is_better * h2_index + !fit_is_better * best_hit_h2;
        }
      }
    }

    // We have a best fit!
    // Fill in t, ONLY in case the best fit is acceptable
    if (ttf_condition) {
      if (best_fit != MAX_FLOAT) {
        // Update the tracks to follow, we'll have to follow up
        // this track on the next iteration :)
        t.hits[t.hitsNum++] = best_hit_h2;

        // Update the track in the bag
        if (t.hitsNum <= 4) {
          // If it is a track made out of less than or equal than 4 hits,
          // we have to allocate it in the tracks pointer
          trackno = atomic_add(tracks_insertPointer, 1);
        }

        // Copy the track into tracks
        tracks[trackno] = t;

        // Add the tracks to the bag of tracks to_follow
        const unsigned int ttfP = atomic_add(ttf_insertPointer, 1) % TTF_MODULO;
        tracks_to_follow[ttfP] = trackno;
      }
      // A track just skipped a module
      // We keep it for another round
      else if (skipped_modules <= MAX_SKIPPED_MODULES) {
        // Form the new mask
        trackno = ((skipped_modules + 1) << 28) | (fulltrackno & 0x8FFFFFFF);

        // Add the tracks to the bag of tracks to_follow
        const unsigned int ttfP = atomic_add(ttf_insertPointer, 1) % TTF_MODULO;
        tracks_to_follow[ttfP] = trackno;
      }
      // If there are only three hits in this track,
      // mark it as "doubtful"
      else if (t.hitsNum == 3) {
        const unsigned int weakP = atomic_add(weaktracks_insertPointer, 1);
        weak_tracks[weakP] = (sensor_id << 24) | trackno;
      }
      // In the "else" case, we couldn't follow up the track,
      // so we won't be track following it anymore.
    }
  }
}

__kernel void clTrackForwarding(__global struct Track* const dev_tracks_per_sensor, __global const char* const dev_input,
  __global int* const dev_tracks_to_follow, __global bool* const dev_hit_used,
  __global int* const dev_atomicsStorage, __global struct Track* const dev_tracklets,
  __global int* const dev_weak_tracks, __global int* const dev_event_offsets,
  __global int* const dev_hit_offsets, __global float* const dev_best_fits,
  __global int* const dev_hit_candidates, __global int* const dev_hit_h2_candidates) {
  
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
  const int hit_offset = sensor_id * MAX_TRACKS_PER_SENSOR;
  __global bool* const hit_used = dev_hit_used;
  __global int* const hit_candidates = dev_hit_candidates;
  __global int* const hit_h2_candidates = dev_hit_h2_candidates;

  __global int* const tracks_to_follow = dev_tracks_to_follow + sensor_id * TTF_MODULO;
  __global struct Track* const tracklets = dev_tracklets + hit_offset;
  __global float* const best_fits = dev_best_fits + sensor_id * blockDim_product;

  __global unsigned int* const tracks_insertPointer = (__global unsigned int*) dev_atomicsStorage;
  __global unsigned int* const weaktracks_insertPointer = (__global unsigned int*) dev_atomicsStorage + 1;
  __global int* const weak_tracks = dev_weak_tracks;
  __global struct Track* const tracks_per_sensor = dev_tracks_per_sensor + sensor_id * MAX_TRACKS_PER_SENSOR;

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

  const int cond_sh_hit_mult = min((int) get_local_size(1), SH_HIT_MULT);
  const int blockDim_sh_hit = NUMTHREADS_X * cond_sh_hit_mult;

  // Let's do the Track Forwarding sequentially now
  unsigned int prev_ttf, last_ttf = 0, diff_ttf = 1;
  int first_sensor = 50 - sensor_id;

  while (first_sensor >= 4 && diff_ttf > 0) {

    // Update sensor data
    if (get_local_id(0) < 6 && get_local_id(1) == 0) {
      const int sensor_number = first_sensor - (get_local_id(0) % 3) * 2;
      __global const int* const sensor_pointer = get_local_id(0) < 3 ? sensor_hitStarts : sensor_hitNums;
      sensor_data[get_local_id(0)] = sensor_pointer[sensor_number];
    }

    prev_ttf = last_ttf;
    last_ttf = ttf_insertPointer[0];
    diff_ttf = last_ttf - prev_ttf;

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    // 2a. Track forwarding
    trackForwarding(
#if USE_SHARED_FOR_HITS
      (__local float*) &sh_hit_x[0],
      (__local float*) &sh_hit_y[0],
      (__local float*) &sh_hit_z[0],
#endif
      hit_Xs,
      hit_Ys,
      hit_Zs,
      hit_used,
      tracks_per_sensor_insertPointer,
      ttf_insertPointer,
      weaktracks_insertPointer,
      blockDim_sh_hit,
      (__local int*) &sensor_data[0],
      diff_ttf,
      blockDim_product,
      tracks_to_follow,
      weak_tracks,
      prev_ttf,
      tracklets,
      tracks_per_sensor,
      sensor_id);

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    first_sensor--;
  }

  // Process the last bunch of track_to_follows
  prev_ttf = last_ttf;
  last_ttf = ttf_insertPointer[0];
  diff_ttf = last_ttf - prev_ttf;

  for (int i=0; i<(diff_ttf + blockDim_product - 1) / blockDim_product; ++i) {
    const unsigned int ttf_element = blockDim_product * i + get_local_id(1) * get_local_size(0) + get_local_id(0);

    if (ttf_element < diff_ttf) {
      const int fulltrackno = tracks_to_follow[prev_ttf + ttf_element];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      const int trackno = fulltrackno & 0x0FFFFFFF;

      // Here we are only interested in three-hit tracks,
      // to mark them as "doubtful"
      if (track_flag) {
        const unsigned int weakP = atomic_add(weaktracks_insertPointer, 1);
        weak_tracks[weakP] = (sensor_id << 24) | trackno;
      }
    }
  }
}
