
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

#define MAX_TRACKS 12000
#define MAX_TRACK_SIZE 24

#define REQUIRED_UNIQUES 0.6f
#define MIN_HITS_TRACK 3
#define MAX_FLOAT FLT_MAX
#define MIN_FLOAT -FLT_MAX
#define MAX_SKIPPED_MODULES 3
#define TTF_MODULO 2000

#define PARAM_W 3966.94f // 0.050 / sqrt( 12. )
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

/**
 * @brief Fills dev_hit_candidates.
 * 
 * @param hit_candidates    
 * @param hit_h2_candidates 
 * @param number_of_sensors 
 * @param sensor_hitStarts  
 * @param sensor_hitNums    
 * @param hit_Xs            
 * @param hit_Ys            
 * @param hit_Zs            
 * @param sensor_Zs         
 */
void fillCandidates(__global int* const hit_candidates,
  __global int* const hit_h2_candidates, const int number_of_sensors,
  __global const int* const sensor_hitStarts, __global const int* const sensor_hitNums,
  __global const float* const hit_Xs, __global const float* const hit_Ys,
  __global const float* const hit_Zs, __global const int* sensor_Zs) {

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
                ASSERT(2 * h0_index < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

                hit_candidates[2 * h0_index] = h1_index;
                first_h1_found = true;
              }
              // The last one, only if the first one has already been found
              else if (first_h1_found && !tol_condition) {
                ASSERT(2 * h0_index + 1 < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

                hit_candidates[2 * h0_index + 1] = h1_index;
                last_h1_found = true;
              }
            }

            if (process_h2_candidates && !last_h2_found) {
              if (!first_h2_found && h1.x > xmin_h2) {
                ASSERT(2 * h0_index < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

                hit_h2_candidates[2 * h0_index] = h1_index;
                first_h2_found = true;
              }
              else if (first_h2_found && h1.x > xmax_h2) {
                ASSERT(2 * h0_index + 1 < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

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
          ASSERT(2 * h0_index + 1 < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

          hit_candidates[2 * h0_index + 1] = hitstarts_s2 + hitnums_s2;
        }

        if (process_h2_candidates && first_h2_found && !last_h2_found) {
          ASSERT(2 * h0_index + 1 < 2 * (sensor_hitStarts[number_of_sensors-1] + sensor_hitNums[number_of_sensors-1]))

          hit_h2_candidates[2 * h0_index + 1] = hitstarts_s2 + hitnums_s2;
        }
      }
    }
  }
}

/**
 * Just fillCandidates
 */
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

  fillCandidates(dev_hit_candidates, dev_hit_h2_candidates, number_of_sensors, sensor_hitStarts, sensor_hitNums,
    hit_Xs, hit_Ys, hit_Zs, sensor_Zs);
}


/**
 * @brief Fits hits to tracks.
 * @details In case the tolerances constraints are met,
 *          returns the chi2 weight of the track. Otherwise,
 *          returns MAX_FLOAT.
 * 
 * @param tx 
 * @param ty 
 * @param h0 
 * @param h1_z
 * @param h2 
 * @return 
 */
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

/**
 * @brief Performs the track forwarding.
 *
 * @param hit_Xs           
 * @param hit_Ys           
 * @param hit_Zs           
 * @param sensor_data      
 * @param sh_hit_x         
 * @param sh_hit_y         
 * @param sh_hit_z         
 * @param diff_ttf         
 * @param blockDim_product 
 * @param tracks_to_follow 
 * @param weak_tracks      
 * @param prev_ttf         
 * @param tracklets        
 * @param tracks           
 * @param number_of_hits   
 */
void trackForwarding(
#if USE_SHARED_FOR_HITS
  __local float* const sh_hit_x, __local float* const sh_hit_y, __local float* const sh_hit_z,
#endif
  __global const float* const hit_Xs, __global const float* const hit_Ys, __global const float* const hit_Zs,
  __global bool* const hit_used, __global int* const tracks_insertPointer,
  __global int* const weaktracks_insertPointer, const int blockDim_sh_hit, __local int* const sensor_data,
  const int blockDim_product,
  __global int* const weak_tracks, __global struct Track* const tracklets,
  __global struct Track* const tracks, const int number_of_hits,
  __global int* const r_tracks_to_follow, __global int* const r_ttf_insertPointer,
  __global int* const w_tracks_to_follow, __global int* const w_ttf_insertPointer) {

  const int read_ttf = r_ttf_insertPointer[0];

  for (int i=0; i<(read_ttf + blockDim_product - 1) / blockDim_product; ++i) {
    const unsigned int ttf_element = blockDim_product * i + get_local_id(1) * get_local_size(0) + get_local_id(0);

    // These variables need to go here, shared memory and scope requirements
    float tx, ty, h1_z;
    unsigned int trackno, fulltrackno, skipped_modules, best_hit_h2;
    struct Track t;
    struct Hit h0;

    // The logic is broken in two parts for shared memory loading
    const bool ttf_condition = ttf_element < read_ttf;
    if (ttf_condition) {
      fulltrackno = r_tracks_to_follow[ttf_element];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      skipped_modules = (fulltrackno & 0x70000000) >> 28;
      trackno = fulltrackno & 0x0FFFFFFF;

      __global const struct Track* const track_pointer = track_flag ? tracklets : tracks;
      
      ASSERT(track_pointer==tracklets ? trackno < number_of_hits : true)
      ASSERT(track_pointer==tracks ? trackno < MAX_TRACKS : true)
      t = track_pointer[trackno];

      // Load last two hits in h0, h1
      const int t_hitsNum = t.hitsNum;
      ASSERT(t_hitsNum < MAX_TRACK_SIZE)
      const int h0_num = t.hits[t_hitsNum - 2];
      const int h1_num = t.hits[t_hitsNum - 1];

      ASSERT(h0_num < number_of_hits)
      h0.x = hit_Xs[h0_num];
      h0.y = hit_Ys[h0_num];
      h0.z = hit_Zs[h0_num];

      ASSERT(h1_num < number_of_hits)
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
        ASSERT(tid < blockDim_sh_hit)
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
        // Mark h2 as used
        ASSERT(best_hit_h2 < number_of_hits)
        hit_used[best_hit_h2] = true;

        // Update the tracks to follow, we'll have to follow up
        // this track on the next iteration :)
        ASSERT(t.hitsNum < MAX_TRACK_SIZE)
        t.hits[t.hitsNum++] = best_hit_h2;

        // Update the track in the bag
        if (t.hitsNum <= 4) {
          ASSERT(t.hits[0] < number_of_hits)
          ASSERT(t.hits[1] < number_of_hits)
          ASSERT(t.hits[2] < number_of_hits)

          // Also mark the first three as used
          hit_used[t.hits[0]] = true;
          hit_used[t.hits[1]] = true;
          hit_used[t.hits[2]] = true;

          // If it is a track made out of less than or equal than 4 hits,
          // we have to allocate it in the tracks pointer
          trackno = atomic_add(tracks_insertPointer, 1);
        }

        // Copy the track into tracks
        ASSERT(trackno < number_of_hits)
        tracks[trackno] = t;

        // Add the tracks to the bag of tracks to_follow
        const unsigned int ttfP = atomic_add(w_ttf_insertPointer, 1);
        w_tracks_to_follow[ttfP] = trackno;
      }
      // A track just skipped a module
      // We keep it for another round
      else if (skipped_modules <= MAX_SKIPPED_MODULES) {
        // Form the new mask
        trackno = ((skipped_modules + 1) << 28) | (fulltrackno & 0x8FFFFFFF);

        // Add the tracks to the bag of tracks to_follow
        const unsigned int ttfP = atomic_add(w_ttf_insertPointer, 1);
        w_tracks_to_follow[ttfP] = trackno;
      }
      // If there are only three hits in this track,
      // mark it as "doubtful"
      else if (t.hitsNum == 3) {
        const unsigned int weakP = atomic_add(weaktracks_insertPointer, 1);
        ASSERT(weakP < number_of_hits)
        weak_tracks[weakP] = trackno;
      }
      // In the "else" case, we couldn't follow up the track,
      // so we won't be track following it anymore.
    }
  }
}

/**
 * @brief Track Creation
 * 
 * @param hit_Xs                  
 * @param hit_Ys                  
 * @param hit_Zs                  
 * @param sensor_data             
 * @param hit_candidates          
 * @param max_numhits_to_process  
 * @param sh_hit_x                
 * @param sh_hit_y                
 * @param sh_hit_z                
 * @param sh_hit_process          
 * @param hit_h2_candidates       
 * @param blockDim_sh_hit         
 * @param best_fits               
 * @param tracklets_insertPointer 
 * @param ttf_insertPointer       
 * @param tracklets               
 * @param tracks_to_follow        
 */
void trackCreation(
#if USE_SHARED_FOR_HITS
  __local float* const sh_hit_x, __local float* const sh_hit_y, __local float* const sh_hit_z,
#endif
  __global const float* const hit_Xs, __global const float* const hit_Ys, __global const float* const hit_Zs,
  __local int* const sensor_data, __global int* const hit_candidates, __global int* const max_numhits_to_process,
  __global int* const hit_h2_candidates, const int blockDim_sh_hit, __global float* const best_fits,
  __global int* const tracklets_insertPointer, __global int* const ttf_insertPointer,
  __global struct Track* const tracklets, __global int* const tracks_to_follow,
  const int h0_index, bool inside_bounds) {

  // Track creation starts
  unsigned int best_hit_h1, best_hit_h2;
  struct Hit h0, h1;
  int first_h1, first_h2, last_h2;
  float dymax;

  // bool inside_bounds = inside_bounds;
  unsigned int num_h1_to_process = 0;
  float best_fit = MAX_FLOAT;

  // We will repeat this for performance reasons
  if (inside_bounds) {
    h0.x = hit_Xs[h0_index];
    h0.y = hit_Ys[h0_index];
    h0.z = hit_Zs[h0_index];
    
    // Calculate new dymax
    const float s1_z = hit_Zs[sensor_data[1]];
    const float h_dist = fabs(s1_z - h0.z);
    dymax = PARAM_MAXYSLOPE * h_dist;

    // Only iterate in the hits indicated by hit_candidates :)
    first_h1 = hit_candidates[2 * h0_index];
    const int last_h1 = hit_candidates[2 * h0_index + 1];
    num_h1_to_process = last_h1 - first_h1;
    atomic_max(max_numhits_to_process, num_h1_to_process);
    ASSERT(max_numhits_to_process[0] >= num_h1_to_process)
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  // Only iterate max_numhits_to_process[0] iterations (with get_local_size(1) threads) :D :D :D
  for (int j=0; j<(max_numhits_to_process[0] + get_local_size(1) - 1) / get_local_size(1); ++j) {
    const int h1_element = get_local_size(1) * j + get_local_id(1);
    inside_bounds &= h1_element < num_h1_to_process;
    int h1_index;
    float dz_inverted;

    if (inside_bounds) {
      h1_index = first_h1 + h1_element;
      h1.x = hit_Xs[h1_index];
      h1.y = hit_Ys[h1_index];
      h1.z = hit_Zs[h1_index];

      dz_inverted = 1.f / (h1.z - h0.z);
      
      first_h2 = hit_h2_candidates[2 * h1_index];
      last_h2 = hit_h2_candidates[2 * h1_index + 1];
      // In case there be no h2 to process,
      // we can preemptively prevent further processing
      inside_bounds &= first_h2 != -1;
    }

    // Iterate in the third list of hits
    // Tiled memory access on h2
    for (int k=0; k<(sensor_data[SENSOR_DATA_HITNUMS + 2] + blockDim_sh_hit - 1) / blockDim_sh_hit; ++k) {

#if USE_SHARED_FOR_HITS
      barrier(CLK_LOCAL_MEM_FENCE);
      if (get_local_id(1) < SH_HIT_MULT) {
        const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
        const int sh_hit_no = blockDim_sh_hit * k + tid;
        if (sh_hit_no < sensor_data[SENSOR_DATA_HITNUMS + 2]) {
          const int h2_index = sensor_data[2] + sh_hit_no;

          // Coalesced memory accesses
          ASSERT(tid < blockDim_sh_hit)
          sh_hit_x[tid] = hit_Xs[h2_index];
          sh_hit_y[tid] = hit_Ys[h2_index];
          sh_hit_z[tid] = hit_Zs[h2_index];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

      if (inside_bounds) {

        const int last_hit_h2 = min(blockDim_sh_hit * (k + 1), sensor_data[SENSOR_DATA_HITNUMS + 2]);
        for (int kk=blockDim_sh_hit * k; kk<last_hit_h2; ++kk) {

          const int h2_index = sensor_data[2] + kk;
          if (h2_index >= first_h2 && h2_index < last_h2) {
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
              best_fit = fit_is_better * fit + !fit_is_better * best_fit;
              best_hit_h1 = fit_is_better * (h1_index) + !fit_is_better * best_hit_h1;
              best_hit_h2 = fit_is_better * (h2_index) + !fit_is_better * best_hit_h2;
            }
          }
        }
      }
    }
  }

  // Compare / Mix the results from the get_local_size(1) threads
  ASSERT(get_local_id(0) * get_local_size(1) + get_local_id(1) < get_local_size(0) * MAX_NUMTHREADS_Y)
  best_fits[get_local_id(0) * get_local_size(1) + get_local_id(1)] = best_fit;

  barrier(CLK_GLOBAL_MEM_FENCE);

  bool accept_track = false;
  if (h0_index != -1 && best_fit != MAX_FLOAT) {
    best_fit = MAX_FLOAT;
    int threadIdx_y_winner = -1;
    for (int i=0; i<get_local_size(1); ++i) {
      const float fit = best_fits[get_local_id(0) * get_local_size(1) + i];
      if (fit < best_fit) {
        best_fit = fit;
        threadIdx_y_winner = i;
      }
    }
    accept_track = get_local_id(1) == threadIdx_y_winner;
  }

  // We have a best fit! - haven't we?
  // Only go through the tracks on the selected thread
  if (accept_track) {
    // Fill in track information

    // Add the track to the bag of tracks
    const unsigned int trackP = atomic_add(tracklets_insertPointer, 1);
    ASSERT(trackP < number_of_hits)
    tracklets[trackP].hitsNum = 3;
    __global unsigned int* const t_hits = tracklets[trackP].hits;
    t_hits[0] = h0_index;
    t_hits[1] = best_hit_h1;
    t_hits[2] = best_hit_h2;

    // Add the tracks to the bag of tracks to_follow
    // Note: The first bit flag marks this is a tracklet (hitsNum == 3),
    // and hence it is stored in tracklets
    const unsigned int ttfP = atomic_add(ttf_insertPointer, 1) % TTF_MODULO;
    tracks_to_follow[ttfP] = 0x80000000 | trackP;
  }
}

/**
 * @brief Track following algorithm, loosely based on Pr/PrPixel.
 * @details It should be simplistic in its design, as is the Pixel VELO problem ;)
 *          Triplets are chosen based on a fit and forwarded using a typical track following algo.
 *          Ghosts are inherently out of the equation, as the algorithm considers all possible
 *          triplets and keeps the best. Upon following, if a hit is not found in the adjacent
 *          module, the track[let] is considered complete.
 *          Clones are removed based off a used-hit mechanism. A global array keeps track of
 *          used hits when forming tracks consist of 4 or more hits.
 *
 *          The algorithm consists in two stages: Track following, and seeding. In each step [iteration],
 *          the track following is performed first, hits are marked as used, and then the seeding is performed,
 *          requiring the first two hits in the triplet to be unused.
 * 
 * @param dev_tracks            
 * @param dev_input             
 * @param dev_tracks_to_follow  
 * @param dev_hit_used          
 * @param dev_atomicsStorage    
 * @param dev_tracklets         
 * @param dev_weak_tracks       
 * @param dev_event_offsets     
 * @param dev_hit_offsets       
 * @param dev_best_fits         
 * @param dev_hit_candidates    
 * @param dev_hit_h2_candidates 
 */
__kernel void clSearchByTriplets(__global struct Track* const dev_tracks, __global const char* const dev_input,
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
  const int hit_offset = sensor_id * 100; // MAX_TRACKS_PER_SENSOR;
  __global bool* const hit_used = dev_hit_used;
  __global int* const hit_candidates = dev_hit_candidates;
  __global int* const hit_h2_candidates = dev_hit_h2_candidates;

  __global int* const tracks_to_follow = dev_tracks_to_follow + sensor_id * TTF_MODULO;
  __global int* const weak_tracks = dev_weak_tracks + hit_offset;
  __global struct Track* const tracklets = dev_tracklets + hit_offset;
  __global float* const best_fits = dev_best_fits + sensor_id * blockDim_product;

  // Initialize variables according to event number and sensor side
  // Insert pointers (atomics)
  int shift = 2;
  __global int* const ttf_insertPointer = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const tracklets_insertPointer = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const sh_hit_lastPointer = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const max_numhits_to_process = (__global int*) dev_atomicsStorage + shift + sensor_id;

  // The fun begins
#if USE_SHARED_FOR_HITS
  __local float sh_hit_x [NUMTHREADS_X * SH_HIT_MULT];
  __local float sh_hit_y [NUMTHREADS_X * SH_HIT_MULT];
  __local float sh_hit_z [NUMTHREADS_X * SH_HIT_MULT];
#endif
  __local int sensor_data [6];

  const int cond_sh_hit_mult = min((int) get_local_size(1), SH_HIT_MULT);
  const int blockDim_sh_hit = NUMTHREADS_X * cond_sh_hit_mult;


  // Start on a different sensor depending on the block
  int first_sensor = 51 - sensor_id;
  
  // Load common things from sensors into shared memory
  if (get_local_id(0) < 6 && get_local_id(1) == 0) {
    const int sensor_number = first_sensor - (get_local_id(0) % 3) * 2;
    __global const int* const sensor_pointer = get_local_id(0) < 3 ? sensor_hitStarts : sensor_hitNums;

    sensor_data[get_local_id(0)] = sensor_pointer[sensor_number];
  }
  else if (get_local_id(0) == 6 && get_local_id(1) == 0) {
    sh_hit_lastPointer[0] = 0;
  }
  else if (get_local_id(0) == 7 && get_local_id(1) == 0) {
    max_numhits_to_process[0] = 0;
  }

  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  // Seeding - Track creation

  // Iterate in all hits for current sensor
  for (int i=0; i<(sensor_hitNums[first_sensor] + blockDim_product - 1) / blockDim_product; ++i) {
    
    // Track creation for hit i
    const int element = blockDim_product * i + get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int h0_index = sensor_hitStarts[first_sensor] + element;
    const bool inside_bounds = element < sensor_hitNums[first_sensor];

    trackCreation(
#if USE_SHARED_FOR_HITS
      (__local float*) &sh_hit_x[0], (__local float*) &sh_hit_y[0], (__local float*) &sh_hit_z[0],
#endif
      hit_Xs, hit_Ys, hit_Zs,
      (__local int*) &sensor_data[0], hit_candidates, max_numhits_to_process,
      hit_h2_candidates, blockDim_sh_hit, best_fits,
      dev_atomicsStorage, ttf_insertPointer, dev_tracks, tracks_to_follow,
      h0_index, inside_bounds);
  }
}



__kernel void clTrackForwarding(__global struct Track* const dev_tracks, __global const char* const dev_input,
  __global int* const dev_tracks_to_follow, __global bool* const dev_hit_used,
  __global int* const dev_atomicsStorage, __global struct Track* const dev_tracklets,
  __global int* const dev_weak_tracks, __global int* const dev_event_offsets,
  __global int* const dev_hit_offsets, __global float* const dev_best_fits,
  __global int* const dev_hit_candidates, __global int* const dev_hit_h2_candidates) {
  
  // Data initialization
  // Each event is treated with two blocks, one for each side.
  const int sensor_id = get_group_id(0);
  // const int sensors_under_process = get_num_groups(0);
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

  // Per event datatypes
  __global struct Track* tracks = dev_tracks;
  __global unsigned int* const tracks_insertPointer = (__global unsigned int*) dev_atomicsStorage;
  __global int* const weaktracks_insertPointer = (__global int*) dev_atomicsStorage + 1;

  // Per side datatypes
  __global bool* const hit_used = dev_hit_used;
  __global int* const hit_candidates = dev_hit_candidates;
  __global int* const hit_h2_candidates = dev_hit_h2_candidates;

  __global int* const tracks_to_follow = dev_tracks_to_follow + sensor_id * TTF_MODULO;
  __global int* const weak_tracks = dev_weak_tracks;
  __global struct Track* const tracklets = dev_tracklets;

  // Initialize variables according to event number and sensor side
  // Insert pointers (atomics)
  int shift = 2;
  __global int* const ttf_insertPointer = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const tracklets_insertPointer = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const sh_hit_lastPointer = (__global int*) dev_atomicsStorage + shift + sensor_id; shift += number_of_sensors;
  __global int* const max_numhits_to_process = (__global int*) dev_atomicsStorage + shift + sensor_id;

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
  for (int first_sensor=51; first_sensor>=5; --first_sensor) {

    // Update pointers
    if (get_local_id(0) < 6 && get_local_id(1) == 0) {
      const int sensor_number = first_sensor - (get_local_id(0) % 3) * 2;
      __global const int* const sensor_pointer = get_local_id(0) < 3 ? sensor_hitStarts : sensor_hitNums;
      sensor_data[get_local_id(0)] = sensor_pointer[sensor_number];
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    // 2a. Track forwarding
    trackForwarding(
#if USE_SHARED_FOR_HITS
      (__local float*) &sh_hit_x[0], (__local float*) &sh_hit_y[0], (__local float*) &sh_hit_z[0],
#endif
      hit_Xs, hit_Ys, hit_Zs, hit_used,
      tracks_insertPointer, weaktracks_insertPointer,
      blockDim_sh_hit, (__local int*) &sensor_data[0],
      blockDim_product, weak_tracks,
      tracklets, tracks, number_of_hits,
      &tracks_to_follow[(first_sensor-4)*TTF_MODULO], &ttf_insertPointer[first_sensor-4],
      &tracks_to_follow[(first_sensor-5)*TTF_MODULO], &ttf_insertPointer[first_sensor-5]);
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  // Process the last bunch of track_to_follows
  const int read_ttf = ttf_insertPointer[4];
  for (int i=0; i<(read_ttf + blockDim_product - 1) / blockDim_product; ++i) {
    const unsigned int ttf_element = blockDim_product * i + get_local_id(1) * get_local_size(0) + get_local_id(0);

    if (ttf_element < read_ttf) {
      const int fulltrackno = (tracks_to_follow + 4*TTF_MODULO)[ttf_element];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      const int trackno = fulltrackno & 0x0FFFFFFF;

      // Here we are only interested in three-hit tracks,
      // to mark them as "doubtful"
      if (track_flag) {
        const unsigned int weakP = atomic_add(weaktracks_insertPointer, 1);
        ASSERT(weakP < number_of_hits)
        weak_tracks[weakP] = trackno;
      }
    }
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  // Compute the three-hit tracks left
  const unsigned int weaktracks_total = weaktracks_insertPointer[0];
  for (int i=0; i<(weaktracks_total + blockDim_product - 1) / blockDim_product; ++i) {
    const unsigned int weaktrack_no = blockDim_product * i + get_local_id(1) * get_local_size(0) + get_local_id(0);
    if (weaktrack_no < weaktracks_total) {

      // Load the tracks from the tracklets
      const struct Track t = tracklets[weak_tracks[weaktrack_no]];

      // Store them in the tracks bag iff they
      // are made out of three unused hits
      if (!hit_used[t.hits[0]] &&
          !hit_used[t.hits[1]] &&
          !hit_used[t.hits[2]]) {
        const unsigned int trackno = atomic_add(tracks_insertPointer, 1);
        ASSERT(trackno < MAX_TRACKS)
        tracks[trackno] = t;
      }
    }
  }
}
