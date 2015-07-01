#include "KernelInvoker.h"

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
  
  // #if USE_SHARED_FOR_HITS
  //   cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  // #else
  //   cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  // #endif
  //   cudaDeviceProp* device_properties = (cudaDeviceProp*) malloc(sizeof(cudaDeviceProp));
  //   cudaGetDeviceProperties(device_properties, 0);

  // Some startup settings
  //   dim3 numBlocks(eventsToProcess);
  //   dim3 numThreads(NUMTHREADS_X, 2);
  //   cudaFuncSetCacheConfig(searchByTriplet, cudaFuncCachePreferShared);
  size_t global_work_size[1] = { eventsToProcess };
  size_t local_work_size[2] = { NUMTHREADS_X, 2 };

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
  
  clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices));

  if (numDevices == 0) {
    std::cout << "No GPU device available." << std::endl;
    std::cout << "Choosing CPU as default device." << std::endl;
    clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices));
    devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
    clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL));
  }
  else {
    devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
    clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL));
  }

  // Step 3: Create context
  cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);

  // Step 4: Creating command queue associate with the context
  cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

  // Step 5: Create program object
  const char* filename = "Kernel.cl";
  std::string sourceStr;
  clCheck(convertClToString(filename, sourceStr));
  const char* source = sourceStr.c_str();
  size_t sourceSize[] = { sourceStr.size() };
  cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
  
  // Step 6: Build program
  clCheck(clBuildProgram(program, 1, devices, NULL, NULL, NULL));

  // Step 7: Memory
  
  // Allocate memory
  // Prepare event offset and hit offset
  std::vector<int> event_offsets;
  std::vector<int> hit_offsets;
  int acc_size = 0, acc_hits = 0;
  for (int i=0; i<eventsToProcess; ++i){
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
  //   cudaCheck(cudaMalloc((void**)&dev_tracks, eventsToProcess * MAX_TRACKS * sizeof(Track)));
  //   cudaCheck(cudaMalloc((void**)&dev_tracklets, acc_hits * sizeof(Track)));
  //   cudaCheck(cudaMalloc((void**)&dev_weak_tracks, acc_hits * sizeof(int)));
  //   cudaCheck(cudaMalloc((void**)&dev_tracks_to_follow, eventsToProcess * TTF_MODULO * sizeof(int)));
  //   cudaCheck(cudaMalloc((void**)&dev_atomicsStorage, eventsToProcess * atomic_space * sizeof(int)));
  //   cudaCheck(cudaMalloc((void**)&dev_event_offsets, event_offsets.size() * sizeof(int)));
  //   cudaCheck(cudaMalloc((void**)&dev_hit_offsets, hit_offsets.size() * sizeof(int)));
  //   cudaCheck(cudaMalloc((void**)&dev_hit_used, acc_hits * sizeof(bool)));
  //   cudaCheck(cudaMalloc((void**)&dev_input, acc_size));
  //   cudaCheck(cudaMalloc((void**)&dev_best_fits, eventsToProcess * numThreads.x * MAX_NUMTHREADS_Y * sizeof(float)));
  //   cudaCheck(cudaMalloc((void**)&dev_hit_candidates, 2 * acc_hits * sizeof(int)));
  //   cudaCheck(cudaMalloc((void**)&dev_hit_h2_candidates, 2 * acc_hits * sizeof(int)));

  // Copy stuff from host memory to GPU buffers
  //   cudaCheck(cudaMemcpy(dev_event_offsets, &event_offsets[0], event_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
  //   cudaCheck(cudaMemcpy(dev_hit_offsets, &hit_offsets[0], hit_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

  cl_mem dev_tracks = clCreateBuffer(context, CL_MEM_READ_WRITE, eventsToProcess * MAX_TRACKS * sizeof(Track), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_tracklets = clCreateBuffer(context, CL_MEM_READ_WRITE, acc_hits * sizeof(Track), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_weak_tracks = clCreateBuffer(context, CL_MEM_READ_WRITE, acc_hits * sizeof(int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_tracks_to_follow = clCreateBuffer(context, CL_MEM_READ_WRITE, eventsToProcess * TTF_MODULO * sizeof(int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_atomicsStorage = clCreateBuffer(context, CL_MEM_READ_WRITE, eventsToProcess * atomic_space * sizeof(int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_event_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, event_offsets.size() * sizeof(int), event_offsets, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, hit_offsets.size() * sizeof(int), hit_offsets, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_used = clCreateBuffer(context, CL_MEM_READ_WRITE, acc_hits * sizeof(bool), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_input = clCreateBuffer(context, CL_MEM_READ_ONLY, acc_size * sizeof(char), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_best_fits = clCreateBuffer(context, CL_MEM_READ_WRITE, eventsToProcess * numThreads.x * MAX_NUMTHREADS_Y * sizeof(float), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_candidates = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * acc_hits * sizeof(int), NULL, &errcode_ret); checkClError(errcode_ret);
  cl_mem dev_hit_h2_candidates = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * acc_hits * sizeof(int), NULL, &errcode_ret); checkClError(errcode_ret);

  acc_size = 0;
  for (int i=0; i<eventsToProcess; ++i){
    // cudaCheck(cudaMemcpy(&dev_input[acc_size], &(*(input[startingEvent + i]))[0], input[startingEvent + i]->size(), cudaMemcpyHostToDevice));
    clCheck(clEnqueueWriteBuffer(commandQueue, &dev_input[acc_size], CL_TRUE, 0, input[startingEvent + i]->size(), &(*(input[startingEvent + i]))[0], 0, NULL, NULL));
    acc_size += input[startingEvent + i]->size();
  }

  // Step 8: Create kernel object
  cl_kernel kernel = clCreateKernel(program, "helloworld", NULL);

  // Step 9: Sets Kernel arguments 
  clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &inputBuffer));
  clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &outputBuffer));
  
  // Step 10: Running the kernel
  size_t global_work_size[1] = { input_strlen };
  clCheck(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL));

  // Step 11: Read the cout put back to host memory
  clCheck(clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, input_strlen * sizeof(char), opencl_output, 0, NULL, NULL));
  
  opencl_output[input_strlen] = '\0'; //Add the terminal character to the end of opencl_output.
  std::cout << std::endl << "opencl_output string:" << std::endl;
  std::cout << opencl_output << std::endl;

  // Step 12: Clean the resources
  clCheck(clReleaseKernel(kernel));       //Release kernel.
  clCheck(clReleaseProgram(program));       //Release the program object.
  clCheck(clReleaseMemObject(inputBuffer));   //Release mem object.
  clCheck(clReleaseMemObject(outputBuffer));
  clCheck(clReleaseCommandQueue(commandQueue)); //Release  Command queue.
  clCheck(clReleaseContext(context));       //Release context.

  if (opencl_output != NULL) {
    free(opencl_output);
    opencl_output = NULL;
  }

  if (devices != NULL) {
    free(devices);
    devices = NULL;
  }

  std::cout << "Passed!" << std::endl;

//   // Adding timing
//   // Timing calculation
//   unsigned int niterations = 1;
//   unsigned int nexperiments = 1;

//   std::vector<std::vector<float>> time_values {nexperiments};
//   std::vector<std::map<std::string, float>> mresults {nexperiments};
//   // std::vector<std::string> exp_names {nexperiments};

//   DEBUG << "Now, on your " << device_properties->name << ": searchByTriplet with " << eventsToProcess << " event" << (eventsToProcess>1 ? "s" : "") << std::endl 
// 	  << " " << nexperiments << " experiments, " << niterations << " iterations" << std::endl;

//   for (auto i=0; i<nexperiments; ++i) {

//     DEBUG << i << ": " << std::flush;

//     if (nexperiments!=1) numThreads.y = i+1;

//     for (auto j=0; j<niterations; ++j) {
//       // Initialize what we need
//       cudaCheck(cudaMemset(dev_hit_used, false, acc_hits * sizeof(bool)));
//       cudaCheck(cudaMemset(dev_atomicsStorage, 0, eventsToProcess * atomic_space * sizeof(int)));
//       cudaCheck(cudaMemset(dev_hit_candidates, -1, 2 * acc_hits * sizeof(int)));
//       cudaCheck(cudaMemset(dev_hit_h2_candidates, -1, 2 * acc_hits * sizeof(int)));

//       // Just for debugging purposes
//       cudaCheck(cudaMemset(dev_tracks, 0, eventsToProcess * MAX_TRACKS * sizeof(Track)));
//       cudaCheck(cudaMemset(dev_tracklets, 0, acc_hits * sizeof(Track)));
//       cudaCheck(cudaMemset(dev_tracks_to_follow, 0, eventsToProcess * TTF_MODULO * sizeof(int)));

//       // searchByTriplet
//       cudaEvent_t start_searchByTriplet, stop_searchByTriplet;
//       float t0;

//       cudaEventCreate(&start_searchByTriplet);
//       cudaEventCreate(&stop_searchByTriplet);

//       cudaEventRecord(start_searchByTriplet, 0 );
      
//       // Dynamic allocation - , 3 * numThreads.x * sizeof(float)
//       searchByTriplet<<<numBlocks, numThreads>>>(dev_tracks, (const char*) dev_input,
//         dev_tracks_to_follow, dev_hit_used, dev_atomicsStorage, dev_tracklets,
//         dev_weak_tracks, dev_event_offsets, dev_hit_offsets, dev_best_fits,
//         dev_hit_candidates, dev_hit_h2_candidates);

//       cudaEventRecord( stop_searchByTriplet, 0 );
//       cudaEventSynchronize( stop_searchByTriplet );
//       cudaEventElapsedTime( &t0, start_searchByTriplet, stop_searchByTriplet );

//       cudaEventDestroy( start_searchByTriplet );
//       cudaEventDestroy( stop_searchByTriplet );

//       cudaCheck( cudaPeekAtLastError() );

//       time_values[i].push_back(t0);

//       DEBUG << "." << std::flush;
//     }

//     DEBUG << std::endl;
//   }

//   // Get results
//   DEBUG << "Number of tracks found per event:" << std::endl << " ";
//   cudaCheck(cudaMemcpy(atomics, dev_atomicsStorage, eventsToProcess * atomic_space * sizeof(int), cudaMemcpyDeviceToHost));
//   for (int i=0; i<eventsToProcess; ++i){
//     const int numberOfTracks = atomics[i];
//     DEBUG << numberOfTracks << ", ";
    
//     output[startingEvent + i].resize(numberOfTracks * sizeof(Track));
//     cudaCheck(cudaMemcpy(&(output[startingEvent + i])[0], &dev_tracks[i * MAX_TRACKS], numberOfTracks * sizeof(Track), cudaMemcpyDeviceToHost));
//   }
//   DEBUG << std::endl;

//   // cudaCheck(cudaMemcpy(hit_candidates, dev_hit_candidates, 2 * acc_hits * sizeof(int), cudaMemcpyDeviceToHost));
//   // std::ofstream hc0("hit_candidates.0");
//   // std::ofstream hc1("hit_candidates.1");
//   // for (int i=0; i<hit_offsets[1] * 2; ++i) hc0 << hit_candidates[i] << std::endl;
//   // for (int i=hit_offsets[1] * 2; i<acc_hits * 2; ++i) hc1 << hit_candidates[i] << std::endl;
//   // hc0.close();
//   // hc1.close();

//   // Print solution tracks of event 0
//   if (PRINT_SOLUTION) {
//     const int numberOfTracks = output[0].size() / sizeof(Track);
//     Track* tracks_in_solution = (Track*) &(output[0])[0];
//     if (logger::ll.verbosityLevel > 0){
//       for(int i=0; i<numberOfTracks; ++i){
//         printTrack(tracks_in_solution, i, zhit_to_module);
//       }
//     }
//   }

//   DEBUG << std::endl << "Time averages:" << std::endl;
//   for (auto i=0; i<nexperiments; ++i){
//     mresults[i] = calcResults(time_values[i]);
//     DEBUG << " nthreads (" << NUMTHREADS_X << ", " << (nexperiments==1 ? numThreads.y : i+1) <<  "): " << mresults[i]["mean"]
//       << " ms (std dev " << mresults[i]["deviation"] << ")" << std::endl;
//   }

//   free(atomics);

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
