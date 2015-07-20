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

  std::map<int, int> zhit_to_module;
  if (logger::ll.verbosityLevel > 0){
    // map to convert from z of hit to module
    for(int i=0; i<number_of_sensors; ++i){
      const int z = h_sensor_Zs[i];
      zhit_to_module[z] = i;
    }

    // Some hits z may not correspond to a sensor's,
    // but be close enough
    for(int i=0; i<*h_no_hits; ++i){
      const int z = h_hit_Zs[i];
      if (zhit_to_module.find(z) == zhit_to_module.end()){
        const int sensor = findClosestModule(z, zhit_to_module);
        zhit_to_module[z] = sensor;
      }
    }
  }

  // Choose which GPU to run on, change this on a multi-GPU system.
  const int device_number = 0;
  
  // Startup settings
  size_t global_work_size[2] = { (size_t) NUMTHREADS_X * eventsToProcess, 2 };
  size_t local_work_size[2] = { (size_t) NUMTHREADS_X, 2 };
  cl_uint work_dim = 2;

  // Step 1: Getting platforms and choose an available one
  cl_uint numPlatforms; // the NO. of platforms
  cl_platform_id platform = NULL; // the chosen platform
  clCheck(clGetPlatformIDs(0, NULL, &numPlatforms));

  // For clarity, choose the first available platform.
  if(numPlatforms > 0)
  {
    cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
    clCheck(clGetPlatformIDs(numPlatforms, platforms, NULL));
    platform = platforms[0];
    free(platforms);
  }

  // Step 2: Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device
  cl_uint       numDevices = 0;
  cl_device_id        *devices;
  
  // Run on CPU - Don't try to poll for GPU devices
  // DEBUG << "Using CPU as device" << std::endl;
  // clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices));
  // devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
  // clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL));

  // We don't check this call. If it fails, we use a CPU device
  cl_int status_ids = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  if (numDevices == 0) {
    DEBUG << "No GPU device available." << std::endl;
    DEBUG << "Choosing CPU as default device." << std::endl;
    clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices));
    devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
    clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL));
  }
  else {
    DEBUG << "Choosing GPU device" << std::endl;
    devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
    clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL));
  }

  // Step 3: Create context
  cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);

  // Step 4: Creating command queue associate with the context
  cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);

  // Step 5: Create program object - KernelDefinitions.h + Kernel.cl
  std::string definitions_str, kernel_str, source_str;
  clCheck(convertClToString("KernelDefinitions.h", definitions_str));
  clCheck(convertClToString("Kernel.cl", kernel_str));
  source_str = definitions_str + kernel_str;
  const char* source = source_str.c_str();
  size_t sourceSize[] = { source_str.size() };
  cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
  
  // Step 6: Build program
  const char* buildOptions = ""; // "-cl-nv-maxrregcount=32";
  cl_int status = clBuildProgram(program, 1, devices, buildOptions, NULL, NULL);

  if (status != CL_SUCCESS) {
    std::cerr << "Error string: " << getErrorString(status) << std::endl;

    if (status == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      char* log = (char *) malloc(log_size);
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
      std::cerr << "Build log: " << std::endl << log << std::endl;
    }

    exit(-1);
  }

  size_t size;
  clCheck(clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG , 0, NULL, &size));

  // Step 7: Memory
  
  // Allocate memory
  // Prepare event offset and hit offset
  std::vector<int> event_offsets;
  std::vector<int> hit_offsets;
  int acc_size = 0, acc_hits = 0;
  for (int i=0; i<eventsToProcess; ++i) {
    EventBeginning* event = (EventBeginning*) &(*(input[startingEvent + i]))[0];
    const int event_size = input[startingEvent + i]->size();

    event_offsets.push_back(acc_size);
    hit_offsets.push_back(acc_hits);

    acc_size += event_size;
    acc_hits += event->numberOfHits;
  }
  
  // Allocate CPU buffers
  const int atomic_space = NUM_ATOMICS + 1;
  int* atomics = (int*) malloc(eventsToProcess * atomic_space * sizeof(int));  
  int* hit_candidates = (int*) malloc(2 * acc_hits * sizeof(int));

  // Allocate GPU buffers
  cl_mem dev_tracks = clCreateBuffer(context, CL_MEM_READ_WRITE, eventsToProcess * MAX_TRACKS * sizeof(Track), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_tracklets = clCreateBuffer(context, CL_MEM_READ_WRITE, acc_hits * sizeof(Track), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_weak_tracks = clCreateBuffer(context, CL_MEM_READ_WRITE, acc_hits * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_tracks_to_follow = clCreateBuffer(context, CL_MEM_READ_WRITE, eventsToProcess * TTF_MODULO * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_atomicsStorage = clCreateBuffer(context, CL_MEM_READ_WRITE, eventsToProcess * atomic_space * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_event_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, event_offsets.size() * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, hit_offsets.size() * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_used = clCreateBuffer(context, CL_MEM_READ_WRITE, acc_hits * sizeof(cl_bool), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_input = clCreateBuffer(context, CL_MEM_READ_ONLY, acc_size * sizeof(cl_char), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_best_fits = clCreateBuffer(context, CL_MEM_READ_WRITE, eventsToProcess * NUMTHREADS_X * MAX_NUMTHREADS_Y * sizeof(cl_float), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_candidates = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * acc_hits * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_h2_candidates = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * acc_hits * sizeof(cl_int), NULL, &errcode_ret); checkClError(errcode_ret);

  clCheck(clEnqueueWriteBuffer(commandQueue, dev_event_offsets, CL_TRUE, 0, event_offsets.size() * sizeof(cl_int), &event_offsets[0], 0, NULL, NULL));
  clCheck(clEnqueueWriteBuffer(commandQueue, dev_hit_offsets, CL_TRUE, 0, hit_offsets.size() * sizeof(cl_int), &hit_offsets[0], 0, NULL, NULL));

  acc_size = 0;
  for (int i=0; i<eventsToProcess; ++i) {
    clCheck(clEnqueueWriteBuffer(commandQueue, dev_input, CL_TRUE, acc_size, input[startingEvent + i]->size(), &(*(input[startingEvent + i]))[0], 0, NULL, NULL));
    acc_size += input[startingEvent + i]->size();
  }

  // Step 8: Create kernel object
  cl_kernel kernel = clCreateKernel(program, "clSearchByTriplets", NULL);

  // Step 9: Sets Kernel arguments 
  clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &dev_tracks));
  clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &dev_input));
  clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &dev_tracks_to_follow));
  clCheck(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &dev_hit_used));
  clCheck(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &dev_atomicsStorage));
  clCheck(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &dev_tracklets));
  clCheck(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &dev_weak_tracks));
  clCheck(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *) &dev_event_offsets));
  clCheck(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *) &dev_hit_offsets));
  clCheck(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *) &dev_best_fits));
  clCheck(clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *) &dev_hit_candidates));
  clCheck(clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *) &dev_hit_h2_candidates));
  
  // Adding timing
  // Timing calculation
  unsigned int niterations = 4;
  unsigned int nexperiments = 4;

  std::vector<std::vector<float>> time_values {nexperiments};
  std::vector<std::map<std::string, float>> mresults {nexperiments};

  // Get and log the OpenCL device ID's
  char cBuffer[1024];
  clCheck( clGetPlatformInfo (devices[0], CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL) );
  DEBUG << "Invoking kernels on your " << cBuffer << " device :)" << std::endl;

  for (auto i=0; i<nexperiments; ++i) {
    // Update the number of threads in Y if more than 1 experiment
    if (nexperiments!=1) {
      size_t wsize_exp = i+1;
      if (i+1 >= 5) {
        // Starting from 5, powers of 2
        wsize_exp = pow(2, i-1);
      }

      global_work_size[1] = wsize_exp;
      local_work_size[1] = wsize_exp;

      DEBUG << wsize_exp << ": " << std::flush;
    }

    for (auto j=0; j<niterations; ++j) {
      // Initialize values to zero
      clInitializeValue<cl_bool>(commandQueue, dev_hit_used, acc_hits, false);
      clInitializeValue<cl_int>(commandQueue, dev_atomicsStorage, eventsToProcess * atomic_space, 0);
      clInitializeValue<cl_int>(commandQueue, dev_hit_candidates, 2 * acc_hits, -1);
      clInitializeValue<cl_int>(commandQueue, dev_hit_h2_candidates, 2 * acc_hits, -1);

      // Just for debugging
      clInitializeValue<cl_char>(commandQueue, dev_tracks, eventsToProcess * MAX_TRACKS * sizeof(Track), 0);
      clInitializeValue<cl_char>(commandQueue, dev_tracklets, acc_hits * sizeof(Track), 0);
      clInitializeValue<cl_int>(commandQueue, dev_tracks_to_follow, eventsToProcess * TTF_MODULO, 0);

      cl_event kernelEvent;

      clCheck(clEnqueueNDRangeKernel(commandQueue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, &kernelEvent));
      clCheck(clFinish(commandQueue));
  
      // Start and end of event
      unsigned long tstart = 0;
      unsigned long tend = 0;
      clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong) , &tstart, NULL);
      clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL);
      clReleaseEvent(kernelEvent);

      // Compute the duration in nanoseconds
      unsigned long tduration = tend - tstart;
      
      // DEBUG << "Execution time (ms): " << tduration / 1000000.0 << std::endl;
      time_values[i].push_back(tduration / 1000000.0f);

      DEBUG << "." << std::flush;
    }
    DEBUG << std::endl;
  }

  // Step 11: Get results
  if (PRINT_SOLUTION) DEBUG << "Number of tracks found per event:" << std::endl << " ";
  clCheck(clEnqueueReadBuffer(commandQueue, dev_atomicsStorage, CL_TRUE, 0, eventsToProcess * atomic_space * sizeof(int), atomics, 0, NULL, NULL));
  for (int i=0; i<eventsToProcess; ++i){
    const int numberOfTracks = atomics[i];
    if (PRINT_SOLUTION) DEBUG << numberOfTracks << ", ";
    
    output[startingEvent + i].resize(numberOfTracks * sizeof(Track));
    if (numberOfTracks > 0) {
      clCheck(clEnqueueReadBuffer(commandQueue, dev_tracks, CL_TRUE, i * MAX_TRACKS, numberOfTracks * sizeof(Track), &(output[startingEvent + i])[0], 0, NULL, NULL));
    }
  }
  if (PRINT_SOLUTION) DEBUG << std::endl;
  
  // Print solution tracks of event 0
  if (PRINT_VERBOSE) {
    const int numberOfTracks = output[0].size() / sizeof(Track);
    Track* tracks_in_solution = (Track*) &(output[0])[0];
    if (logger::ll.verbosityLevel > 0){
      for(int i=0; i<numberOfTracks; ++i){
        printTrack(tracks_in_solution, i, zhit_to_module);
      }
    }
  }

  DEBUG << std::endl << "Time averages:" << std::endl;
  for (auto i=0; i<nexperiments; ++i){
    mresults[i] = calcResults(time_values[i]);
    DEBUG << " nthreads (" << NUMTHREADS_X << ", " << (nexperiments==1 ? local_work_size[1] : i+1) <<  "): " << mresults[i]["mean"]
      << " ms (std dev " << mresults[i]["deviation"] << ")" << std::endl;
  }

  // Step 12: Clean the resources
  clCheck(clReleaseKernel(kernel));
  clCheck(clReleaseProgram(program));
  clCheck(clReleaseCommandQueue(commandQueue));
  clCheck(clReleaseContext(context));
  clCheck(clReleaseMemObject(dev_tracks));
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

  if (devices != NULL) {
    free(devices);
    devices = NULL;
  }

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
void printTrack(Track* tracks, const int trackNumber, const std::map<int, int>& zhit_to_module){
  const Track t = tracks[trackNumber];
  DEBUG << "Track #" << trackNumber << ", length " << (int) t.hitsNum << std::endl;

  for(int i=0; i<t.hitsNum; ++i){
    const int hitNumber = t.hits[i];
    const unsigned int id = h_hit_IDs[hitNumber];
    const float x = h_hit_Xs[hitNumber];
    const float y = h_hit_Ys[hitNumber];
    const float z = h_hit_Zs[hitNumber];
    const int module = zhit_to_module.at((int) z);

    DEBUG << " " << std::setw(8) << id << " (" << hitNumber << ")"
      << " module " << std::setw(2) << module
      << ", x " << std::setw(6) << x
      << ", y " << std::setw(6) << y
      << ", z " << std::setw(6) << z << std::endl;
  }

  DEBUG << std::endl;
}

/**
 * The z of the hit may not correspond to any z in the sensors.
 * @param  z              
 * @param  zhit_to_module 
 * @return                sensor number
 */
int findClosestModule(const int z, const std::map<int, int>& zhit_to_module){
  if (zhit_to_module.find(z) != zhit_to_module.end())
    return zhit_to_module.at(z);

  int error = 0;
  while(true){
    error++;
    const int lowerAttempt = z - error;
    const int higherAttempt = z + error;

    if (zhit_to_module.find(lowerAttempt) != zhit_to_module.end()){
      return zhit_to_module.at(lowerAttempt);
    }
    if (zhit_to_module.find(higherAttempt) != zhit_to_module.end()){
      return zhit_to_module.at(higherAttempt);
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
