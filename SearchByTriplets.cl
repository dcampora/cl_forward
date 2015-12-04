
void trackCreation(
  __global const float* const hit_Xs,
  __global const float* const hit_Ys,
  __global const float* const hit_Zs,
  __local int* const sensor_data,
  __global int* const hit_candidates,
  __global int* const max_numhits_to_process,
  __global int* const hit_h2_candidates,
  __global int* const best_fits,
  __global int* const tracklets_insertPointer,
  __global struct CL_Track* const tracklets,
  __global int* const tracks_to_follow,
  const int h0_index,
  bool inside_bounds) {

  // CL_Track creation starts
  unsigned int best_hit_h1, best_hit_h2;
  struct CL_Hit h0, h1;
  int first_h1, first_h2, last_h2;
  float dymax;

  unsigned int num_h1_to_process = 0;
  float best_fit = MAX_FLOAT;
  bool best_fit_found = false;

  // We will repeat this for performance reasons
  if (inside_bounds) {
    h0.x = hit_Xs[h0_index];
    h0.y = hit_Ys[h0_index];
    h0.z = hit_Zs[h0_index];
    
    // Calculate new dymax
    const float s1_z = hit_Zs[sensor_data[1]];
    const float h_dist = fabs(s1_z - h0.z);
    dymax = PARAM_MAXYSLOPE * h_dist;

    // Only iterate in the hits indicated by hit_candidates
    first_h1 = hit_candidates[2 * h0_index];
    const int last_h1 = hit_candidates[2 * h0_index + 1];
    num_h1_to_process = last_h1 - first_h1;
    atomic_max(max_numhits_to_process, num_h1_to_process);
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (inside_bounds) {

    // Only iterate max_numhits_to_process[0] iterations (with get_local_size(1) threads)
    for (int j=0; j<(max_numhits_to_process[0] + get_local_size(1) - 1) / get_local_size(1); ++j) {
      const int h1_element = get_local_size(1) * j + get_local_id(1);
      const bool h1_inside_bounds = h1_element < num_h1_to_process;

      if (h1_inside_bounds) {
        const int h1_index = first_h1 + h1_element;
        h1.x = hit_Xs[h1_index];
        h1.y = hit_Ys[h1_index];
        h1.z = hit_Zs[h1_index];

        const float dz_inverted = 1.f / (h1.z - h0.z);
        
        first_h2 = hit_h2_candidates[2 * h1_index];
        last_h2 = hit_h2_candidates[2 * h1_index + 1];
        // In case there be no h2 to process,
        // we can preemptively prevent further processing
        const bool h2_to_process = first_h2 != -1;

        // Iterate in the third list of hits
        if (h2_to_process) {
          for (int k=0; k<sensor_data[SENSOR_DATA_HITNUMS + 2]; ++k) {
            const int h2_index = sensor_data[2] + k;

            if (h2_index >= first_h2 && h2_index < last_h2) {
              struct CL_Hit h2;
              h2.x = hit_Xs[h2_index];
              h2.y = hit_Ys[h2_index];
              h2.z = hit_Zs[h2_index];

              // Predictions of x and y for this hit
              const float z2_tz = (h2.z - h0.z) * dz_inverted;
              const float x = h0.x + (h1.x - h0.x) * z2_tz;
              const float y = h0.y + (h1.y - h0.y) * z2_tz;
              const float dx = x - h2.x;
              const float dy = y - h2.y;

              if (fabs(h1.y - h0.y) < dymax && fabs(dx) < PARAM_TOLERANCE && fabs(dy) < PARAM_TOLERANCE) {
                // Calculate fit
                const float scatterNum = (dx * dx) + (dy * dy);
                const float scatterDenom = 1.f / (h2.z - h1.z);
                const float scatter = scatterNum * scatterDenom * scatterDenom;
                const bool condition = scatter < MAX_SCATTER;
                const float fit = condition * scatter + !condition * MAX_FLOAT; 

                const bool fit_is_better = fit < best_fit;
                best_fit_found |= fit_is_better;

                best_fit = fit_is_better * fit + !fit_is_better * best_fit;
                best_hit_h1 = fit_is_better * (h1_index) + !fit_is_better * best_hit_h1;
                best_hit_h2 = fit_is_better * (h2_index) + !fit_is_better * best_hit_h2;
              }
            }
          }
        }
      }
    }
  }

  // Compare / Mix the results from the get_local_size(1) threads
  const int val_best_fit = *((int*) &best_fit);
  const int old_best_fit = best_fit_found ? atomic_min(best_fits + h0_index, val_best_fit) : 0;
  barrier(CLK_GLOBAL_MEM_FENCE);
  const int new_best_fit = best_fit_found ? best_fits[h0_index] : 0;

  const bool accept_track = (h0_index != -1) && best_fit_found &&
    (old_best_fit != val_best_fit) && (new_best_fit == val_best_fit);

  // We have a best fit! - haven't we?
  // Only go through the tracks on the selected thread
  if (accept_track) {
    // Fill in track information

    // Add the track to the bag of tracks
    const unsigned int trackP = atomic_add(tracklets_insertPointer, 1);
    tracklets[trackP].hitsNum = 3;
    __global unsigned int* const t_hits = tracklets[trackP].hits;
    t_hits[0] = h0_index;
    t_hits[1] = best_hit_h1;
    t_hits[2] = best_hit_h2;
  }
}

/**
 * @brief CL_Track forwarding algorithm, loosely based on Pr/PrPixel.
 * @details It should be simplistic in its design, as is the Pixel VELO problem ;)
 *          Triplets are chosen based on a fit and forwarded using a typical track forwarding algo.
 *          Ghosts are inherently out of the equation, as the algorithm considers all possible
 *          triplets and keeps the best. Upon forwarding, if a hit is not found in the adjacent
 *          module, the track[let] is considered complete.
 *          Clones are removed based off a used-hit mechanism. A global array keeps track of
 *          used hits when forming tracks consist of 4 or more hits.
 */
__kernel void clSearchByTriplets(__global struct CL_Track* const dev_tracks, __global const char* const dev_input,
  __global int* const dev_tracks_to_follow, __global bool* const dev_hit_used,
  __global int* const dev_atomicsStorage, __global struct CL_Track* const dev_tracklets,
  __global int* const dev_weak_tracks, __global int* const dev_event_offsets,
  __global int* const dev_hit_offsets, __global int* const dev_best_fits,
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
  __global struct CL_Track* const tracklets = dev_tracklets + hit_offset;
  __global int* const best_fits = dev_best_fits;

  __global unsigned int* const tracks_insertPointer = (__global unsigned int*) dev_atomicsStorage;
  __global unsigned int* const weaktracks_insertPointer = (__global unsigned int*) dev_atomicsStorage + 1;
  __global int* const weak_tracks = dev_weak_tracks;
  __global struct CL_Track* const tracks = dev_tracks;

  // Initialize variables according to event number and sensor side
  // Insert pointers (atomics)
  int shift = 2;
  __global int* const tracklets_insertPointer = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const max_numhits_to_process = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const tracks_per_sensor_insertPointer = (__global int*) dev_atomicsStorage + shift + sensor_id;

  // The fun begins
  __local int sensor_data [6];

  // Start on a different sensor depending on the block
  int first_sensor = 51 - sensor_id;
  
  // Load common things from sensors into shared memory
  if (get_local_id(0) < 6 && get_local_id(1) == 0) {
    const int sensor_number = first_sensor - (get_local_id(0) % 3) * 2;
    __global const int* const sensor_pointer = get_local_id(0) < 3 ? sensor_hitStarts : sensor_hitNums;

    sensor_data[get_local_id(0)] = sensor_pointer[sensor_number];
  }
  else if (get_local_id(0) == 6 && get_local_id(1) == 0) {
    max_numhits_to_process[0] = 0;
  }

  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  // Seeding - CL_Track creation
  // Iterate in all hits for current sensor
  for (int i=0; i<(sensor_hitNums[first_sensor] + get_local_size(0) - 1) / get_local_size(0); ++i) {
    
    // CL_Track creation for hit i
    const int element = get_local_size(0) * i + get_local_id(0);
    const int h0_index = sensor_hitStarts[first_sensor] + element;
    const bool inside_bounds = element < sensor_hitNums[first_sensor];

    trackCreation(
      hit_Xs,
      hit_Ys,
      hit_Zs,
      (__local int*) &sensor_data[0],
      hit_candidates,
      max_numhits_to_process,
      hit_h2_candidates,
      best_fits,
      tracklets_insertPointer,
      tracklets,
      tracks_to_follow,
      h0_index,
      inside_bounds);
  }
}
