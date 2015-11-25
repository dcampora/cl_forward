
__kernel void clFillCandidates(__global struct Track* const dev_tracks, __global const char* const dev_input,
  __global int* const dev_tracks_to_follow, __global bool* const dev_hit_used,
  __global int* const dev_atomicsStorage, __global struct Track* const dev_tracklets,
  __global int* const dev_weak_tracks, __global int* const dev_event_offsets,
  __global int* const dev_hit_offsets, __global float* const dev_best_fits,
  __global int* const dev_hit_candidates, __global int* const dev_hit_h2_candidates) {

  // Data initialization
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
  __global int* const hit_candidates = dev_hit_candidates;
  __global int* const hit_h2_candidates = dev_hit_h2_candidates;

  // Let's go
  int first_sensor = get_group_id(0) + 4;
  const int second_sensor = first_sensor - 2;
  const int blockDim_product = get_local_size(0) * get_local_size(1);

  const bool process_h1_candidates = first_sensor >= 4;
  const bool process_h2_candidates = first_sensor <= number_of_sensors - 3;

  // Sensor dependent calculations
  const int z_s0 = process_h2_candidates ? sensor_Zs[first_sensor + 2] : 0;
  const int z_s2 = process_h2_candidates ? sensor_Zs[second_sensor] : 0;

  // Iterate in all hits in z0
  for (int i=0; i<(sensor_hitNums[first_sensor] + blockDim_product - 1) / blockDim_product; ++i) {
    const int h0_element = blockDim_product * i + get_local_id(1) * get_local_size(0) + get_local_id(0);
    bool inside_bounds = h0_element < sensor_hitNums[first_sensor];

    if (inside_bounds) {
      bool first_h1_found = false, last_h1_found = false;
      bool first_h2_found = false, last_h2_found = false;
      const int h0_index = sensor_hitStarts[first_sensor] + h0_element;
      int h1_index;
      struct Hit h0;
      h0.x = hit_Xs[h0_index];
      h0.z = hit_Zs[h0_index];
      const int hitstarts_s2 = sensor_hitStarts[second_sensor];
      const int hitnums_s2 = sensor_hitNums[second_sensor];

      float xmin_h2, xmax_h2;
      if (process_h2_candidates) {
        // Note: Here, we take h0 as if it were h1, the rest
        // of the notation is fine.
        
        // Min and max possible x0s
        const float h_dist = fabs(h0.z - z_s0);
        const float dxmax = PARAM_MAXXSLOPE_CANDIDATES * h_dist;
        const float x0_min = h0.x - dxmax;
        const float x0_max = h0.x + dxmax;

        // Min and max possible h1s for that h0
        float z2_tz = (((float) z_s2 - z_s0)) / (h0.z - z_s0);
        float x = x0_max + (h0.x - x0_max) * z2_tz;
        xmin_h2 = x - PARAM_TOLERANCE_CANDIDATES;

        x = x0_min + (h0.x - x0_min) * z2_tz;
        xmax_h2 = x + PARAM_TOLERANCE_CANDIDATES;
      }
      
      if (first_sensor >= 4) {
        // Iterate in all hits in z1
        for (int h1_element=0; h1_element<hitnums_s2; ++h1_element) {
          inside_bounds = h1_element < hitnums_s2;

          if (inside_bounds) {
            h1_index = hitstarts_s2 + h1_element;
            struct Hit h1;
            h1.x = hit_Xs[h1_index];
            h1.z = hit_Zs[h1_index];

            if (process_h1_candidates && !last_h1_found) {
              // Check if h0 and h1 are compatible
              const float h_dist = fabs(h1.z - h0.z);
              const float dxmax = PARAM_MAXXSLOPE_CANDIDATES * h_dist;
              const bool tol_condition = fabs(h1.x - h0.x) < dxmax;
              
              // Find the first one
              if (!first_h1_found && tol_condition) {
                hit_candidates[2 * h0_index] = h1_index;
                first_h1_found = true;
              }
              // The last one, only if the first one has already been found
              else if (first_h1_found && !tol_condition) {
                hit_candidates[2 * h0_index + 1] = h1_index;
                last_h1_found = true;
              }
            }

            if (process_h2_candidates && !last_h2_found) {
              if (!first_h2_found && h1.x > xmin_h2) {
                hit_h2_candidates[2 * h0_index] = h1_index;
                first_h2_found = true;
              }
              else if (first_h2_found && h1.x > xmax_h2) {
                hit_h2_candidates[2 * h0_index + 1] = h1_index;
                last_h2_found = true;
              }
            }

            if ((!process_h1_candidates || last_h1_found) &&
                (!process_h2_candidates || last_h2_found)) {
              break;
            }
          }
        }

        // Note: If first is not found, then both should be -1
        // and there wouldn't be any iteration
        if (process_h1_candidates && first_h1_found && !last_h1_found) {
          hit_candidates[2 * h0_index + 1] = hitstarts_s2 + hitnums_s2;
        }

        if (process_h2_candidates && first_h2_found && !last_h2_found) {
          hit_h2_candidates[2 * h0_index + 1] = hitstarts_s2 + hitnums_s2;
        }
      }
    }
  }
}
