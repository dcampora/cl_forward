
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
    float deviation = 0.0f, variance = 0.0f, mean = 0.0f, min = MAX_FLOAT, max = 0.0f;

    for(auto it = times.begin(); it != times.end(); it++){
        const float seconds = (*it);
        mean += seconds;
        variance += seconds * seconds;

        if (seconds < min) min = seconds;
        if (seconds > max) max = seconds;
    }

    mean /= times.size();
    variance = (variance / times.size()) - (mean * mean);
    deviation = std::sqrt(variance);

    results["variance"] = variance;
    results["deviation"] = deviation;
    results["mean"] = mean;
    results["min"] = min;
    results["max"] = max;

    return results;
}

void checkClError(const cl_int errcode_ret) {
  // CHECK_OPENCL_ERROR(errcode_ret, "Error ");
  if (errcode_ret != CL_SUCCESS) {
    std::cerr << "Error " << errcode_ret << std::endl;
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
