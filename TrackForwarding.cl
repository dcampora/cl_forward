
float fitHitToTrack(const float tx, const float ty, const struct CL_Hit* h0, const float h1_z, const struct CL_Hit* h2) {
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

__kernel void clTrackForwarding(
  __global struct CL_Track* const dev_tracks_per_sensor,
  __global const char* const dev_input,
  __global int* const dev_atomicsStorage,
  __global struct CL_Track* const dev_tracklets,
  __global int* const dev_weak_tracks,
  __global int* const dev_best_fits,
  __global int* const best_fits_hit_index) {
  
  // Data initialization
  // Each event is treated with two blocks, one for each side.
  const int sensor_id = get_group_id(0);
  const int sensors_under_process = get_num_groups(0);

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

  __global struct CL_Track* const tracklets = dev_tracklets + hit_offset;
  __global int* const best_fits = dev_best_fits + sensor_id * number_of_sensors * number_of_hits;

  __global unsigned int* const weaktracks_insertPointer = (__global unsigned int*) dev_atomicsStorage + 1;
  __global int* const weak_tracks = dev_weak_tracks;
  __global struct CL_Track* const tracks_per_sensor = dev_tracks_per_sensor + sensor_id * MAX_TRACKS_PER_SENSOR;

  // Initialize variables according to event number and sensor side
  // Insert pointers (atomics)
  int shift = 2;
  __global int* const tracklets_insertPointer = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const max_numhits_to_process = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const tracks_per_sensor_insertPointer = (__global int*) dev_atomicsStorage + shift + sensor_id;

  // The fun begins
  struct Track t;
  struct Hit h0, h1;
  float tx, ty, best_fit;
  bool best_fit_found;
  int skipped_modules, first_hit_index, best_hit_h2;

  // Forward all tracklets of the assigned sensor sensor_id
  const int number_of_tracklets = tracklets_insertPointer[0];
  for (int trackletID=0; trackletID<(number_of_tracklets + get_local_size(0) - 1) / get_local_size(0); ++trackletID) {
    const int trackno = trackletID * get_local_size(0) + get_local_id(0);
    const bool tracklet_inside_bounds = trackno < number_of_tracklets;

    if (tracklet_inside_bounds) {
      // Load track
      t = tracklets[trackno];
      skipped_modules = 0;

      // Initialize h0 and h1 with h1 and h2 respectively
      first_hit_index = t.hits[0];
      const int h1_index = t.hits[1];
      const int h2_index = t.hits[2];

      h0.x = hit_Xs[h1_index];
      h0.y = hit_Ys[h1_index];
      h0.z = hit_Zs[h1_index];

      h1.x = hit_Xs[h2_index];
      h1.y = hit_Ys[h2_index];
      h1.z = hit_Zs[h2_index];

      // Track forwarding over t, for all hits in the next module
      // Preload line calculations
      const float td = 1.0f / (h1.z - h0.z);
      const float txn = (h1.x - h0.x);
      const float tyn = (h1.y - h0.y);
      tx = txn * td;
      ty = tyn * td;
    }

    // Iterate over all sensors, and forward tracks
    for (int lookup_sensor = 51 - sensor_id - 5; lookup_sensor >= 0; --lookup_sensor) {
      // Initialize fit machinery
      best_fit_found = false;
      best_fit = MAX_FLOAT;

      if (tracklet_inside_bounds && skipped_modules <= MAX_SKIPPED_MODULES + 1) {
        // Find best h2 candidate with get_local_size(1) threads
        const int sensor_hitnums = sensor_hitNums[lookup_sensor];
        for (int i=0; i<(sensor_hitnums + get_local_size(1) - 1) / get_local_size(1); ++i) {
          const int h2_element = i * get_local_size(1) + get_local_id(1);
          const bool h2_inside_bounds = h2_element < sensor_hitnums;

          if (h2_inside_bounds) {
            // Load h2
            const int h2_index = sensor_hitStarts[lookup_sensor] + h2_element;

            struct CL_Hit h2;
            h2.x = hit_Xs[h2_index];
            h2.y = hit_Ys[h2_index];
            h2.z = hit_Zs[h2_index];

            // Best fit machinery
            const float fit = fitHitToTrack(tx, ty, &h0, h1.z, &h2);
            const bool fit_is_better = fit < best_fit;
            best_fit_found |= fit_is_better;

            best_fit = fit_is_better * fit + !fit_is_better * best_fit;
            best_hit_h2 = fit_is_better * h2_index + !fit_is_better * best_hit_h2;
          }
        }
      }

      // Compare / Mix the results from the get_local_size(1) threads
      const int val_best_fit = *((int*) &best_fit);
      const int old_best_fit = best_fit_found ? atomic_min(best_fits + lookup_sensor * number_of_hits + first_hit_index, val_best_fit) : 0;
      barrier(CLK_GLOBAL_MEM_FENCE);
      const int new_best_fit = best_fit_found ? best_fits[lookup_sensor * number_of_hits + first_hit_index] : 0;

      const bool accept_forward = best_fit_found &&
        (old_best_fit != val_best_fit) && (new_best_fit == val_best_fit);

      // Communicate to other threads in case the current thread is the best
      if (accept_forward) {
        best_fits_hit_index[lookup_sensor * number_of_hits + first_hit_index] = best_hit_h2;
      }

      barrier(CLK_GLOBAL_MEM_FENCE);

      if (tracklet_inside_bounds < MAX_SKIPPED_MODULES + 1) {
        // We didn't find a hit

        if (*((float*)&new_best_fit) == MAX_FLOAT) {
          if (skipped_modules > MAX_SKIPPED_MODULES) {
            // Increment skipped_modules to distinguish
            // forming track that just skipped MAX_SKIPPED_MODULES from
            // forming track that skipped MAX_SKIPPED_MODULES some iterations before
            skipped_modules++;

            // Cease search for this track, and add it
            // accordingly either to weak_tracks or to tracks

            if (get_local_id(1) == 0) {
              // This is a one-man job...
              if (t.hitsNum == 3) {
                const unsigned int weakP = atomic_add(weaktracks_insertPointer, 1);

                // Keep the info of which sensor_id treated this track for later
                weak_tracks[weakP] = (sensor_id << 24) | trackno;
              }
              else {
                const unsigned int trackP = atomic_add(tracks_per_sensor_insertPointer, 1);
                tracks_per_sensor[trackP] = t;
              }
            }
          }
          else {
            skipped_modules++;

            // Corner case: We just looked up the last sensor
            // Add track accordingly either to weak_tracks or to tracks
            if (lookup_sensor == 0 && get_local_id(0)) {
              if (t.hitsNum == 3) {
                const unsigned int weakP = atomic_add(weaktracks_insertPointer, 1);
                weak_tracks[weakP] = (sensor_id << 24) | trackno;
              }
              else {
                const unsigned int trackP = atomic_add(tracks_per_sensor_insertPointer, 1);
                tracks_per_sensor[trackP] = t;
              }
            }
          }
        }

        else {
          // We found a new hit for our track
          // Update the track
          const int h2_index = best_fits_hit_index[lookup_sensor * number_of_hits + first_hit_index];
          t.hits[t.hitsNum++] = h2_index;

          // Corner case: We just looked up the last sensor
          // Add track accordingly
          // It must have at least 4 hits, since we found one
          if (lookup_sensor == 0 && get_local_id(0)) {
            const unsigned int trackP = atomic_add(tracks_per_sensor_insertPointer, 1);
            tracks_per_sensor[trackP] = t;
          }
          else if (lookup_sensor != 0) {
            // Update skipped_modules
            skipped_modules = 0;

            // Reassign h0
            h0 = h1;

            // Reload the found h2 as the new h1
            h1.x = hit_Xs[h2_index];
            h1.y = hit_Ys[h2_index];
            h1.z = hit_Zs[h2_index];

            // Track forwarding over t, for all hits in the next module
            // Preload line calculations
            const float td = 1.0f / (h1.z - h0.z);
            const float txn = (h1.x - h0.x);
            const float tyn = (h1.y - h0.y);
            tx = txn * td;
            ty = tyn * td;
          }
        }
      }
    }
  }
}
