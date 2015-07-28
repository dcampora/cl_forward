
#include "Tools.h"

// TODO: Remove globals in the short future
int*   h_no_sensors;
int*   h_no_hits;
int*   h_sensor_Zs;
int*   h_sensor_hitStarts;
int*   h_sensor_hitNums;
unsigned int* h_hit_IDs;
float* h_hit_Xs;
float* h_hit_Ys;
float* h_hit_Zs;

/* convert the kernel file into a string */
int convertClToString(const char *filename, std::string& s)
{
  size_t size;
  char*  str;
  std::fstream f(filename, (std::fstream::in | std::fstream::binary));

  if (f.is_open()) {
    size_t fileSize;
    f.seekg(0, std::fstream::end);
    size = fileSize = (size_t)f.tellg();
    f.seekg(0, std::fstream::beg);
    str = new char[size+1];
    
    if (!str) {
      f.close();
      return 0;
    }

    f.read(str, fileSize);
    f.close();
    str[size] = '\0';
    s = str;
    delete[] str;
    return 0;
  }

  std::cout << "Error: failed to open file\n:" << filename << std::endl;
  return -1;
}

void preorder_by_x(std::vector<const std::vector<uint8_t>* > & input) {
  // Order *all* the input vectors by h_hit_Xs natural order
  // per sensor
  const int eventsToProcess = input.size();
  const std::vector<uint8_t>* startingEvent_input = input[0];
  setHPointersFromInput((uint8_t*) &(*startingEvent_input)[0], startingEvent_input->size());

  int number_of_sensors = *h_no_sensors;
  for (int i=0; i<eventsToProcess; ++i) {
    int acc_hitnums = 0;
    const std::vector<uint8_t>* event_input = input[i];
    setHPointersFromInput((uint8_t*) &(*event_input)[0], event_input->size());

    for (int j=0; j<number_of_sensors; j++) {
      const int hitnums = h_sensor_hitNums[j];
      quicksort(h_hit_Xs, h_hit_Ys, h_hit_Zs, h_hit_IDs, acc_hitnums, acc_hitnums + hitnums - 1);
      acc_hitnums += hitnums;
    }
  }
}

void setHPointersFromInput(uint8_t * input, size_t size){
  uint8_t * end = input + size;

  h_no_sensors       = (int32_t*)input; input += sizeof(int32_t);
  h_no_hits          = (int32_t*)input; input += sizeof(int32_t);
  h_sensor_Zs        = (int32_t*)input; input += sizeof(int32_t) * *h_no_sensors;
  h_sensor_hitStarts = (int32_t*)input; input += sizeof(int32_t) * *h_no_sensors;
  h_sensor_hitNums   = (int32_t*)input; input += sizeof(int32_t) * *h_no_sensors;
  h_hit_IDs          = (uint32_t*)input; input += sizeof(uint32_t) * *h_no_hits;
  h_hit_Xs           = (float*)  input; input += sizeof(float)   * *h_no_hits;
  h_hit_Ys           = (float*)  input; input += sizeof(float)   * *h_no_hits;
  h_hit_Zs           = (float*)  input; input += sizeof(float)   * *h_no_hits;

  if (input != end)
    throw std::runtime_error("failed to deserialize event");
}

std::map<std::string, float> calcResults(std::vector<float>& times){
    // sqrt ( E( (X - m)2) )
    std::map<std::string, float> results;
    float deviation = 0.0f, variance = 0.0f, mean = 0.0f, min_value = MAX_FLOAT, max_value = 0.0f;

    for(auto it = times.begin(); it != times.end(); it++){
        const float seconds = (*it);
        mean += seconds;
        variance += seconds * seconds;

        if (seconds < min_value) min_value = seconds;
        if (seconds > max_value) max_value = seconds;
    }

    mean /= times.size();
    variance = (variance / times.size()) - (mean * mean);
    deviation = std::sqrt(variance);

    results["variance"] = variance;
    results["deviation"] = deviation;
    results["mean"] = mean;
    results["min"] = min_value;
    results["max"] = max_value;

    return results;
}

void checkClError(const cl_int errcode_ret) {
  // CHECK_OPENCL_ERROR(errcode_ret, "Error ");
  if (errcode_ret != CL_SUCCESS) {
    std::cerr << "Error " << getErrorString(errcode_ret) << std::endl;
    exit(-1);
  }
}

void quicksort (float* a, float* b, float* c, unsigned int* d, int start, int end) {
    if (start < end) {
        const int pivot = divide(a, b, c, d, start, end);
        quicksort(a, b, c, d, start, pivot - 1);
        quicksort(a, b, c, d, pivot + 1, end);
    }
}

int divide (float* a, float* b, float* c, unsigned int* d, int start, int end) {
    int left;
    int right;
    float pivot;
 
    pivot = a[start];
    left = start;
    right = end;
 
    while (left < right) {
        while (a[right] > pivot) {
            right--;
        }
 
        while ((left < right) && (a[left] <= pivot)) {
            left++;
        }
 
        if (left < right) {
            swap(a[left], a[right]);
            swap(b[left], b[right]);
            swap(c[left], c[right]);
            swap(d[left], d[right]);
        }
    }
 
    swap(a[right], a[start]);
    swap(b[right], b[start]);
    swap(c[right], c[start]);
    swap(d[right], d[start]);
 
    return right;
}

template<typename T>
void swap (T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

const char *getErrorString (cl_int error) {
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

void clChoosePlatform(cl_device_id*& devices, cl_platform_id& platform) {
  // Choose the first available platform
  cl_platform_id* clPlatformIDs;
  cl_uint numPlatforms;
  clCheck(clGetPlatformIDs(0, NULL, &numPlatforms));
  if(numPlatforms > 0)
  {
    cl_platform_id* platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));
    clCheck(clGetPlatformIDs(numPlatforms, platforms, NULL));
    platform = platforms[0];
    free(platforms);
  }

  // Choose a device from the platform according to DEVICE_PREFERENCE
  cl_uint numCpus = 0;
  cl_uint numGpus = 0;
  cl_uint numAccelerators = 0;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numCpus);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numGpus);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &numAccelerators);
  devices = (cl_device_id*) malloc(numAccelerators * sizeof(cl_device_id));

  DEBUG << std::endl << "Devices available: " << std::endl
    << "CPU: " << numCpus << std::endl
    << "GPU: " << numGpus << std::endl
    << "Accelerators: " << numAccelerators << std::endl;

  if (DEVICE_PREFERENCE == DEVICE_CPU && numCpus > 0) {
    DEBUG << "Choosing CPU" << std::endl;
    clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numCpus, devices, NULL));
  }
  else if (DEVICE_PREFERENCE == DEVICE_GPU && numGpus > 0) {
    DEBUG << "Choosing GPU" << std::endl;
    clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numGpus, devices, NULL));
  }
  else if (DEVICE_PREFERENCE == DEVICE_ACCELERATOR && numAccelerators > 0) {
    DEBUG << "Choosing accelerator" << std::endl;
    clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, numAccelerators, devices, NULL));
  }
  else {
    // We couldn't match the preference.
    // Let's try the first device that appears available.
    cl_uint numDevices = 0;
    clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));
    if (numDevices > 0) {
      DEBUG << "Preference device couldn't be met" << std::endl
            << "Choosing an available OpenCL capable device" << std::endl;
      clCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL));
    }
    else {
      DEBUG << "No OpenCL capable device detected" << std::endl
            << "Check the drivers, OpenCL runtime or ICDs are available" << std::endl;
      exit(-1);
    }
  }
  DEBUG << std::endl;
}
