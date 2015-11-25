#include "KernelInvoker.h"
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

extern int*   h_no_sensors;
extern int*   h_no_hits;
extern int*   h_sensor_Zs;
extern int*   h_sensor_hitStarts;
extern int*   h_sensor_hitNums;
extern unsigned int* h_hit_IDs;
extern float* h_hit_Xs;
extern float* h_hit_Ys;
extern float* h_hit_Zs;

int invokeParallelSearch(
    const int startingEvent,
    const int eventsToProcess,
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output) {

  cl_int errcode_ret;
  const std::vector<uint8_t>* startingEvent_input = input[startingEvent];
  setHPointersFromInput((uint8_t*) &(*startingEvent_input)[0], startingEvent_input->size());
  int number_of_sensors = *h_no_sensors;
  
  // Startup settings
  // Now we are going to call with number_of_sensors - 4
  int fillCandidates_blocks = (number_of_sensors - 4);
  size_t fillCandidates_global_work_size[2] = { (size_t) NUMTHREADS_X * fillCandidates_blocks, 4 };
  size_t fillCandidates_local_work_size[2] = { (size_t) NUMTHREADS_X, 4 };
  cl_uint fillCandidates_work_dim = 2;

  int searchByTriplets_blocks = (number_of_sensors - 4);
  size_t searchByTriplets_global_work_size[2] = { (size_t) NUMTHREADS_X * searchByTriplets_blocks, 4 };
  size_t searchByTriplets_local_work_size[2] = { (size_t) NUMTHREADS_X, 4 };
  cl_uint searchByTriplets_work_dim = 2;

  size_t trackForwarding_global_work_size[2] = { (size_t) TF_NUMTHREADS_X * (number_of_sensors - 5), TF_NUMTHREADS_Y };
  size_t trackForwarding_local_work_size[2] = { (size_t) TF_NUMTHREADS_X, TF_NUMTHREADS_Y };
  cl_uint trackForwarding_work_dim = 2;

  size_t cloneKiller_global_work_size[1] = { (size_t) NUMTHREADS_X };
  size_t cloneKiller_local_work_size[1] = { (size_t) NUMTHREADS_X };
  cl_uint cloneKiller_work_dim = 1;

  // Choose platform according to the macro DEVICE_PREFERENCE
  cl_device_id* devices;
  cl_platform_id platform = NULL;
  clChoosePlatform(devices, platform);

  // Step 3: Create context
  cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, &errcode_ret); checkClError(errcode_ret);

  // Step 4: Creating command queue associate with the context
  cl_command_queue commandQueue = clCreateCommandQueue(context, devices[DEVICE_NUMBER], CL_QUEUE_PROFILING_ENABLE, NULL);

  // Step 5: Create program object
  std::vector<std::string> source_files =
    {"TrackForwarding.cl", "SearchByTriplets.cl", "FillCandidates.cl", "CloneKiller.cl"}; // "KernelDefinitions.h"
  std::string source_str = "";
  for (auto s : source_files) {
    std::string temp_str;
    clCheck(convertClToString(s.c_str(), temp_str));
    source_str += temp_str;
  }
  const char* source = source_str.c_str();
  size_t sourceSize[] = { source_str.size() };
  cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
  
  // Step 6: Build program
  // const char* buildOptions = "";
  // const char* buildOptions = "-cl-nv-maxrregcount=32";
  // const char* buildOptions = "-g -s /home/dcampora/nfs/projects/gpu/tf_opencl/KernelDefinitions.cl -s /home/dcampora/nfs/projects/gpu/tf_opencl/kernel_searchByTriplets.cl"; 
  const char* buildOptions = "-g -s \"/home/dcampora/projects/gpu/cl_forward_one_event/cl/TrackForwarding.cl\"";
  cl_int status = clBuildProgram(program, 1, devices, buildOptions, NULL, NULL);

  if (status != CL_SUCCESS) {
    std::cerr << "Error string: " << getErrorString(status) << std::endl;

    if (status == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      clGetProgramBuildInfo(program, devices[DEVICE_NUMBER], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      char* log = (char *) malloc(log_size);
      clGetProgramBuildInfo(program, devices[DEVICE_NUMBER], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
      std::cerr << "Build log: " << std::endl << log << std::endl;
    }

    exit(-1);
  }

  size_t size;
  clCheck(clGetProgramBuildInfo(program, devices[DEVICE_NUMBER], CL_PROGRAM_BUILD_LOG , 0, NULL, &size));

  // Step 7: Memory
  
  // Allocate memory
  // Prepare event offset and hit offset
  std::vector<int> event_offsets;
  std::vector<int> hit_offsets;
  int acc_size = 0, acc_hits = 0;
  
  // For only one event
  EventBeginning* event = (EventBeginning*) &(*(input[startingEvent]))[0];
  const int event_size = input[startingEvent]->size();
  event_offsets.push_back(acc_size);
  hit_offsets.push_back(acc_hits);
  acc_size += event_size;
  acc_hits += event->numberOfHits;
  
  // Allocate CPU buffers
  const int atomic_space = NUM_ATOMICS + 2;
  int* atomics = (int*) malloc(number_of_sensors * atomic_space * sizeof(int));  
  int* hit_candidates = (int*) malloc(2 * acc_hits * sizeof(int));

  // Allocate GPU buffers
  cl_mem dev_tracks = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_TRACKS * sizeof(Track), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_tracks_per_sensor = clCreateBuffer(context, CL_MEM_READ_WRITE, number_of_sensors * MAX_TRACKS_PER_SENSOR * sizeof(Track), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_tracklets = clCreateBuffer(context, CL_MEM_READ_WRITE, number_of_sensors * MAX_TRACKS_PER_SENSOR * sizeof(Track), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_weak_tracks = clCreateBuffer(context, CL_MEM_READ_WRITE, acc_hits * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_tracks_to_follow = clCreateBuffer(context, CL_MEM_READ_WRITE, number_of_sensors * TTF_MODULO * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_atomicsStorage = clCreateBuffer(context, CL_MEM_READ_WRITE, number_of_sensors * atomic_space * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_event_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, event_offsets.size() * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, hit_offsets.size() * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_used = clCreateBuffer(context, CL_MEM_READ_WRITE, acc_hits * sizeof(cl_bool), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_input = clCreateBuffer(context, CL_MEM_READ_ONLY, acc_size * sizeof(cl_char), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_best_fits = clCreateBuffer(context, CL_MEM_READ_WRITE, number_of_sensors * NUMTHREADS_X * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_candidates = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * acc_hits * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_h2_candidates = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * acc_hits * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_best_fits_forwarding = clCreateBuffer(context, CL_MEM_READ_WRITE, number_of_sensors * number_of_sensors * TF_NUMTHREADS_X * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);

  clCheck(clEnqueueWriteBuffer(commandQueue, dev_event_offsets, CL_TRUE, 0, event_offsets.size() * sizeof(cl_int), &event_offsets[0], 0, NULL, NULL));
  clCheck(clEnqueueWriteBuffer(commandQueue, dev_hit_offsets, CL_TRUE, 0, hit_offsets.size() * sizeof(cl_int), &hit_offsets[0], 0, NULL, NULL));

  acc_size = 0;
  // Only one event
  clCheck(clEnqueueWriteBuffer(commandQueue, dev_input, CL_TRUE, acc_size, input[startingEvent]->size(), &(*(input[startingEvent]))[0], 0, NULL, NULL));
  acc_size += input[startingEvent]->size();

  // Step 8: Create kernel_searchByTriplets object
  cl_kernel kernel_fillCandidates = clCreateKernel(program, "clFillCandidates", NULL);
  cl_kernel kernel_searchByTriplets = clCreateKernel(program, "clSearchByTriplets", NULL);
  cl_kernel kernel_trackForwarding = clCreateKernel(program, "clTrackForwarding", NULL);
  cl_kernel kernel_cloneKiller = clCreateKernel(program, "clCloneKiller", NULL);

  // Step 9: Sets kernel_searchByTriplets arguments 
  clCheck(clSetKernelArg(kernel_fillCandidates, 0, sizeof(cl_mem), (void *) &dev_tracks));
  clCheck(clSetKernelArg(kernel_fillCandidates, 1, sizeof(cl_mem), (void *) &dev_input));
  clCheck(clSetKernelArg(kernel_fillCandidates, 2, sizeof(cl_mem), (void *) &dev_tracks_to_follow));
  clCheck(clSetKernelArg(kernel_fillCandidates, 3, sizeof(cl_mem), (void *) &dev_hit_used));
  clCheck(clSetKernelArg(kernel_fillCandidates, 4, sizeof(cl_mem), (void *) &dev_atomicsStorage));
  clCheck(clSetKernelArg(kernel_fillCandidates, 5, sizeof(cl_mem), (void *) &dev_tracklets));
  clCheck(clSetKernelArg(kernel_fillCandidates, 6, sizeof(cl_mem), (void *) &dev_weak_tracks));
  clCheck(clSetKernelArg(kernel_fillCandidates, 7, sizeof(cl_mem), (void *) &dev_event_offsets));
  clCheck(clSetKernelArg(kernel_fillCandidates, 8, sizeof(cl_mem), (void *) &dev_hit_offsets));
  clCheck(clSetKernelArg(kernel_fillCandidates, 9, sizeof(cl_mem), (void *) &dev_best_fits));
  clCheck(clSetKernelArg(kernel_fillCandidates, 10, sizeof(cl_mem), (void *) &dev_hit_candidates));
  clCheck(clSetKernelArg(kernel_fillCandidates, 11, sizeof(cl_mem), (void *) &dev_hit_h2_candidates));

  clCheck(clSetKernelArg(kernel_searchByTriplets, 0, sizeof(cl_mem), (void *) &dev_tracks));
  clCheck(clSetKernelArg(kernel_searchByTriplets, 1, sizeof(cl_mem), (void *) &dev_input));
  clCheck(clSetKernelArg(kernel_searchByTriplets, 2, sizeof(cl_mem), (void *) &dev_tracks_to_follow));
  clCheck(clSetKernelArg(kernel_searchByTriplets, 3, sizeof(cl_mem), (void *) &dev_hit_used));
  clCheck(clSetKernelArg(kernel_searchByTriplets, 4, sizeof(cl_mem), (void *) &dev_atomicsStorage));
  clCheck(clSetKernelArg(kernel_searchByTriplets, 5, sizeof(cl_mem), (void *) &dev_tracklets));
  clCheck(clSetKernelArg(kernel_searchByTriplets, 6, sizeof(cl_mem), (void *) &dev_weak_tracks));
  clCheck(clSetKernelArg(kernel_searchByTriplets, 7, sizeof(cl_mem), (void *) &dev_event_offsets));
  clCheck(clSetKernelArg(kernel_searchByTriplets, 8, sizeof(cl_mem), (void *) &dev_hit_offsets));
  clCheck(clSetKernelArg(kernel_searchByTriplets, 9, sizeof(cl_mem), (void *) &dev_best_fits));
  clCheck(clSetKernelArg(kernel_searchByTriplets, 10, sizeof(cl_mem), (void *) &dev_hit_candidates));
  clCheck(clSetKernelArg(kernel_searchByTriplets, 11, sizeof(cl_mem), (void *) &dev_hit_h2_candidates));

  clCheck(clSetKernelArg(kernel_trackForwarding, 0, sizeof(cl_mem), (void *) &dev_tracks_per_sensor));
  clCheck(clSetKernelArg(kernel_trackForwarding, 1, sizeof(cl_mem), (void *) &dev_input));
  clCheck(clSetKernelArg(kernel_trackForwarding, 2, sizeof(cl_mem), (void *) &dev_tracks_to_follow));
  clCheck(clSetKernelArg(kernel_trackForwarding, 3, sizeof(cl_mem), (void *) &dev_hit_used));
  clCheck(clSetKernelArg(kernel_trackForwarding, 4, sizeof(cl_mem), (void *) &dev_atomicsStorage));
  clCheck(clSetKernelArg(kernel_trackForwarding, 5, sizeof(cl_mem), (void *) &dev_tracklets));
  clCheck(clSetKernelArg(kernel_trackForwarding, 6, sizeof(cl_mem), (void *) &dev_weak_tracks));
  clCheck(clSetKernelArg(kernel_trackForwarding, 7, sizeof(cl_mem), (void *) &dev_event_offsets));
  clCheck(clSetKernelArg(kernel_trackForwarding, 8, sizeof(cl_mem), (void *) &dev_hit_offsets));
  clCheck(clSetKernelArg(kernel_trackForwarding, 9, sizeof(cl_mem), (void *) &dev_best_fits_forwarding));
  clCheck(clSetKernelArg(kernel_trackForwarding, 10, sizeof(cl_mem), (void *) &dev_hit_candidates));
  clCheck(clSetKernelArg(kernel_trackForwarding, 11, sizeof(cl_mem), (void *) &dev_hit_h2_candidates));

  clCheck(clSetKernelArg(kernel_cloneKiller, 0, sizeof(cl_mem), (void *) &dev_tracks));
  clCheck(clSetKernelArg(kernel_cloneKiller, 1, sizeof(cl_mem), (void *) &dev_input));
  clCheck(clSetKernelArg(kernel_cloneKiller, 2, sizeof(cl_mem), (void *) &dev_tracks_to_follow));
  clCheck(clSetKernelArg(kernel_cloneKiller, 3, sizeof(cl_mem), (void *) &dev_hit_used));
  clCheck(clSetKernelArg(kernel_cloneKiller, 4, sizeof(cl_mem), (void *) &dev_atomicsStorage));
  clCheck(clSetKernelArg(kernel_cloneKiller, 5, sizeof(cl_mem), (void *) &dev_tracklets));
  clCheck(clSetKernelArg(kernel_cloneKiller, 6, sizeof(cl_mem), (void *) &dev_weak_tracks));
  clCheck(clSetKernelArg(kernel_cloneKiller, 7, sizeof(cl_mem), (void *) &dev_event_offsets));
  clCheck(clSetKernelArg(kernel_cloneKiller, 8, sizeof(cl_mem), (void *) &dev_hit_offsets));
  clCheck(clSetKernelArg(kernel_cloneKiller, 9, sizeof(cl_mem), (void *) &dev_best_fits));
  clCheck(clSetKernelArg(kernel_cloneKiller, 10, sizeof(cl_mem), (void *) &dev_hit_candidates));
  clCheck(clSetKernelArg(kernel_cloneKiller, 11, sizeof(cl_mem), (void *) &dev_hit_h2_candidates));
  clCheck(clSetKernelArg(kernel_cloneKiller, 12, sizeof(cl_mem), (void *) &dev_tracks_per_sensor));
  
  // Adding timing
  // Timing calculation
  unsigned int niterations = 4;
  unsigned int nexperiments = 4;

  std::vector<std::vector<float>> times_fillCandidates {nexperiments};
  std::vector<std::vector<float>> times_searchByTriplets {nexperiments};
  std::vector<std::vector<float>> times_trackForwarding {nexperiments};
  std::vector<std::vector<float>> times_cloneKiller {nexperiments};
  std::vector<std::map<std::string, float>> mresults_fillCandidates {nexperiments};
  std::vector<std::map<std::string, float>> mresults_searchByTriplets {nexperiments};
  std::vector<std::map<std::string, float>> mresults_trackForwarding {nexperiments};
  std::vector<std::map<std::string, float>> mresults_cloneKiller {nexperiments};

  // Get and log the OpenCL device name
  char deviceName [1024];
  clCheck(clGetDeviceInfo(devices[DEVICE_NUMBER], CL_DEVICE_NAME, 1024, deviceName, NULL));
  DEBUG << "Invoking kernels on your " << deviceName << std::endl;

  // size_t
  // clCheck(clGetDeviceInfo(devices[DEVICE_NUMBER], CL_DEVICE_MAX_WORK_GROUP_SIZE, 1024, deviceName, NULL));
  // DEBUG << "CL_DEVICE_MAX_WORK_GROUP_SIZE " << deviceName << std::endl;

  for (auto i=0; i<nexperiments; ++i) {
    // Update the number of threads in Y if more than 1 experiment
    if (nexperiments!=1) {
      fillCandidates_global_work_size[1] = i+1;
      fillCandidates_local_work_size[1] = i+1;
      searchByTriplets_global_work_size[1] = i+1;
      searchByTriplets_local_work_size[1] = i+1;

      DEBUG << i+1 << ": " << std::flush;
    }

    for (auto j=0; j<niterations; ++j) {
      // Initialize values to zero
      clInitializeValue<cl_bool>(commandQueue, dev_hit_used, acc_hits, false);
      clInitializeValue<cl_int>(commandQueue, dev_atomicsStorage, number_of_sensors * atomic_space, 0);
      clInitializeValue<cl_int>(commandQueue, dev_hit_candidates, 2 * acc_hits, -1);
      clInitializeValue<cl_int>(commandQueue, dev_hit_h2_candidates, 2 * acc_hits, -1);
      clInitializeValue<cl_int>(commandQueue, dev_best_fits, number_of_sensors * NUMTHREADS_X, 0x7FFFFFFF);
      clInitializeValue<cl_int>(commandQueue, dev_best_fits_forwarding, number_of_sensors * number_of_sensors * TF_NUMTHREADS_X, 0x7FFFFFFF);

      // Just for debugging
      clInitializeValue<cl_char>(commandQueue, dev_tracks, MAX_TRACKS * sizeof(Track), 0);
      clInitializeValue<cl_char>(commandQueue, dev_tracklets, acc_hits * sizeof(Track), 0);
      clInitializeValue<cl_int>(commandQueue, dev_tracks_to_follow, number_of_sensors * TTF_MODULO, 0);
      clCheck(clFinish(commandQueue));

      cl_event event_searchByTriplets, event_fillCandidates, event_trackForwarding, event_cloneKiller;

      clCheck(clEnqueueNDRangeKernel(commandQueue, kernel_fillCandidates, fillCandidates_work_dim, NULL, fillCandidates_global_work_size, fillCandidates_local_work_size, 0, NULL, &event_fillCandidates));
      clCheck(clEnqueueNDRangeKernel(commandQueue, kernel_searchByTriplets, searchByTriplets_work_dim, NULL, searchByTriplets_global_work_size, searchByTriplets_local_work_size, 0, NULL, &event_searchByTriplets));
      clCheck(clEnqueueNDRangeKernel(commandQueue, kernel_trackForwarding, trackForwarding_work_dim, NULL, trackForwarding_global_work_size, trackForwarding_local_work_size, 0, NULL, &event_trackForwarding));
      clCheck(clEnqueueNDRangeKernel(commandQueue, kernel_cloneKiller, cloneKiller_work_dim, NULL, cloneKiller_global_work_size, cloneKiller_local_work_size, 0, NULL, &event_cloneKiller));
      clCheck(clFinish(commandQueue));
  
      // Start and end of event
      unsigned long tstart = 0;
      unsigned long tend = 0;

      clGetEventProfilingInfo(event_fillCandidates, CL_PROFILING_COMMAND_START, sizeof(cl_ulong) , &tstart, NULL);
      clGetEventProfilingInfo(event_fillCandidates, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL);
      clReleaseEvent(event_fillCandidates);

      // Compute the duration in nanoseconds
      unsigned long tduration = tend - tstart;
      times_fillCandidates[i].push_back(tduration / 1000000.0f);

      clGetEventProfilingInfo(event_searchByTriplets, CL_PROFILING_COMMAND_START, sizeof(cl_ulong) , &tstart, NULL);
      clGetEventProfilingInfo(event_searchByTriplets, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL);
      clReleaseEvent(event_searchByTriplets);
      tduration = tend - tstart;
      times_searchByTriplets[i].push_back(tduration / 1000000.0f);

      clGetEventProfilingInfo(event_trackForwarding, CL_PROFILING_COMMAND_START, sizeof(cl_ulong) , &tstart, NULL);
      clGetEventProfilingInfo(event_trackForwarding, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL);
      clReleaseEvent(event_trackForwarding);
      tduration = tend - tstart;
      times_trackForwarding[i].push_back(tduration / 1000000.0f);

      clGetEventProfilingInfo(event_cloneKiller, CL_PROFILING_COMMAND_START, sizeof(cl_ulong) , &tstart, NULL);
      clGetEventProfilingInfo(event_cloneKiller, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL);
      clReleaseEvent(event_cloneKiller);
      tduration = tend - tstart;
      times_cloneKiller[i].push_back(tduration / 1000000.0f);

      DEBUG << "." << std::flush;
    }
    DEBUG << std::endl;
  }

  // Step 11: Get results
  if (PRINT_SOLUTION) DEBUG << "Number of tracks found per event:" << std::endl << " ";
  clCheck(clEnqueueReadBuffer(commandQueue, dev_atomicsStorage, CL_TRUE, 0, atomic_space * sizeof(int), atomics, 0, NULL, NULL));
  for (int i=0; i<eventsToProcess; ++i){
    const int numberOfTracks = atomics[i];
    if (PRINT_SOLUTION) DEBUG << numberOfTracks << ", ";
    
    output[startingEvent + i].resize(numberOfTracks * sizeof(Track));
    if (numberOfTracks > 0) {
      clCheck(clEnqueueReadBuffer(commandQueue, dev_tracks, CL_TRUE, i * MAX_TRACKS * sizeof(Track), numberOfTracks * sizeof(Track), &(output[startingEvent + i])[0], 0, NULL, NULL));
    }
  }
  if (PRINT_SOLUTION) DEBUG << std::endl;
  
  if (PRINT_VERBOSE) {
    // Print solution of all events processed, to results
    for (int i=0; i<eventsToProcess; ++i) {

      // Calculate z to sensor map
      std::map<int, int> zhit_to_module;
      setHPointersFromInput((uint8_t*) &(*(input[startingEvent + i]))[0], input[startingEvent + i]->size());
      int number_of_sensors = *h_no_sensors;
      if (logger::ll.verbosityLevel > 0){
        // map to convert from z of hit to module
        for(int j=0; j<number_of_sensors; ++j){
          const int z = h_sensor_Zs[j];
          zhit_to_module[z] = j;
        }
        // Some hits z may not correspond to a sensor's,
        // but be close enough
        for(int j=0; j<*h_no_hits; ++j){
          const int z = (int) h_hit_Zs[j];
          if (zhit_to_module.find(z) == zhit_to_module.end()){
            const int sensor = findClosestModule(z, zhit_to_module);
            zhit_to_module[z] = sensor;
          }
        }
      }

      // Print to output file with event no.
      const int numberOfTracks = output[i].size() / sizeof(Track);
      Track* tracks_in_solution = (Track*) &(output[startingEvent + i])[0];
      std::ofstream outfile (std::string(RESULTS_FOLDER) + std::string("/") + toString(i) + std::string(".out"));
      for(int j=0; j<numberOfTracks; ++j){
        printTrack(tracks_in_solution, j, zhit_to_module, outfile);
      }
      outfile.close();
    }
  }

  DEBUG << std::endl << "Time averages:" << std::endl;
  for (auto i=0; i<nexperiments; ++i){
    mresults_fillCandidates[i] = calcResults(times_fillCandidates[i]);
    mresults_searchByTriplets[i] = calcResults(times_searchByTriplets[i]);
    mresults_trackForwarding[i] = calcResults(times_trackForwarding[i]);
    mresults_cloneKiller[i] = calcResults(times_cloneKiller[i]);

    DEBUG << " nthreads (" << NUMTHREADS_X << ", " << (nexperiments==1 ? fillCandidates_local_work_size[1] : i+1) <<  "):" << std::endl;
    DEBUG << "  fillCandidates: " << mresults_fillCandidates[i]["mean"] << " ms (std dev " << mresults_fillCandidates[i]["deviation"] << ")" << std::endl;
    DEBUG << "  searchByTriplets: " << mresults_searchByTriplets[i]["mean"] << " ms (std dev " << mresults_searchByTriplets[i]["deviation"] << ")" << std::endl;
    DEBUG << "  trackForwarding: " << mresults_trackForwarding[i]["mean"] << " ms (std dev " << mresults_trackForwarding[i]["deviation"] << ")" << std::endl;
    DEBUG << "  cloneKiller: " << mresults_cloneKiller[i]["mean"] << " ms (std dev " << mresults_cloneKiller[i]["deviation"] << ")" << std::endl << std::endl;
  }

  // Step 12: Clean the resources
  clCheck(clReleaseKernel(kernel_fillCandidates));
  clCheck(clReleaseKernel(kernel_searchByTriplets));
  clCheck(clReleaseKernel(kernel_trackForwarding));
  clCheck(clReleaseKernel(kernel_cloneKiller));

  clCheck(clReleaseProgram(program));
  clCheck(clReleaseCommandQueue(commandQueue));
  clCheck(clReleaseContext(context));

  clCheck(clReleaseMemObject(dev_tracks));
  clCheck(clReleaseMemObject(dev_tracks_per_sensor));
  clCheck(clReleaseMemObject(dev_tracklets));
  clCheck(clReleaseMemObject(dev_weak_tracks));
  clCheck(clReleaseMemObject(dev_tracks_to_follow));
  clCheck(clReleaseMemObject(dev_atomicsStorage));
  clCheck(clReleaseMemObject(dev_event_offsets));
  clCheck(clReleaseMemObject(dev_hit_offsets));
  clCheck(clReleaseMemObject(dev_hit_used));
  clCheck(clReleaseMemObject(dev_input));
  clCheck(clReleaseMemObject(dev_best_fits));
  clCheck(clReleaseMemObject(dev_hit_candidates));
  clCheck(clReleaseMemObject(dev_hit_h2_candidates));

  free(atomics);
  free(devices);

  return 0;
}

/**
 * Prints tracks
 * Track #n, length <length>:
 *  <ID> module <module>, x <x>, y <y>, z <z>
 * 
 * @param tracks      
 * @param trackNumber 
 */
void printTrack(Track* tracks, const int trackNumber,
  const std::map<int, int>& zhit_to_module, std::ofstream& outstream){
  const Track t = tracks[trackNumber];
  outstream << "Track #" << trackNumber << ", length " << (int) t.hitsNum << std::endl;

  for(int i=0; i<t.hitsNum; ++i){
    const int hitNumber = t.hits[i];
    const unsigned int id = h_hit_IDs[hitNumber];
    const float x = h_hit_Xs[hitNumber];
    const float y = h_hit_Ys[hitNumber];
    const float z = h_hit_Zs[hitNumber];
    const int module = zhit_to_module.at((int) z);

    outstream << " " << std::setw(8) << id << " (" << hitNumber << ")"
      << " module " << std::setw(2) << module
      << ", x " << std::setw(6) << x
      << ", y " << std::setw(6) << y
      << ", z " << std::setw(6) << z << std::endl;
  }

  outstream << std::endl;
}

/**
 * The z of the hit may not correspond to any z in the sensors.
 * @param  z              
 * @param  zhit_to_module 
 * @return                sensor number
 */
int findClosestModule(const int z, const std::map<int, int>& zhit_to_module){
  auto it = zhit_to_module.find(z);
  if (it != zhit_to_module.end())
    return it->second;

  int error = 0;
  while(true){
    error++;
    const int lowerAttempt = z - error;
    const int higherAttempt = z + error;

    auto it_lowerAttempt = zhit_to_module.find(lowerAttempt);
    if (it_lowerAttempt != zhit_to_module.end()){
      return it_lowerAttempt->second;
    }

    auto it_higherAttempt = zhit_to_module.find(higherAttempt);
    if (it_higherAttempt != zhit_to_module.end()){
      return it_higherAttempt->second;
    }
  }
}

void printOutAllSensorHits(int* prevs, int* nexts){
  DEBUG << "All valid sensor hits: " << std::endl;
  for(int i=0; i<h_no_sensors[0]; ++i){
    for(int j=0; j<h_sensor_hitNums[i]; ++j){
      int hit = h_sensor_hitStarts[i] + j;

      if(nexts[hit] != -1){
        DEBUG << hit << ", " << nexts[hit] << std::endl;
      }
    }
  }
}

void printOutSensorHits(int sensorNumber, int* prevs, int* nexts){
  for(int i=0; i<h_sensor_hitNums[sensorNumber]; ++i){
    int hstart = h_sensor_hitStarts[sensorNumber];

    DEBUG << hstart + i << ": " << prevs[hstart + i] << ", " << nexts[hstart + i] << std::endl;
  }
}

void printInfo(int numberOfSensors, int numberOfHits) {
  numberOfSensors = numberOfSensors>52 ? 52 : numberOfSensors;

  DEBUG << "Read info:" << std::endl
    << " no sensors: " << h_no_sensors[0] << std::endl
    << " no hits: " << h_no_hits[0] << std::endl
    << numberOfSensors << " sensors: " << std::endl;

  for (int i=0; i<numberOfSensors; ++i){
    DEBUG << " Zs: " << h_sensor_Zs[i] << std::endl
      << " hitStarts: " << h_sensor_hitStarts[i] << std::endl
      << " hitNums: " << h_sensor_hitNums[i] << std::endl << std::endl;
  }

  DEBUG << numberOfHits << " hits: " << std::endl;

  for (int i=0; i<numberOfHits; ++i){
    DEBUG << " hit_id: " << h_hit_IDs[i] << std::endl
      << " hit_X: " << h_hit_Xs[i] << std::endl
      << " hit_Y: " << h_hit_Ys[i] << std::endl
      << " hit_Z: " << h_hit_Zs[i] << std::endl << std::endl;
  }
}

void getMaxNumberOfHits(char*& input, int& maxHits){
  int* l_no_sensors = (int*) &input[0];
  int* l_no_hits = (int*) (l_no_sensors + 1);
  int* l_sensor_Zs = (int*) (l_no_hits + 1);
  int* l_sensor_hitStarts = (int*) (l_sensor_Zs + l_no_sensors[0]);
  int* l_sensor_hitNums = (int*) (l_sensor_hitStarts + l_no_sensors[0]);

  maxHits = 0;
  for(int i=0; i<l_no_sensors[0]; ++i){
    if(l_sensor_hitNums[i] > maxHits)
      maxHits = l_sensor_hitNums[i];
  }
}
