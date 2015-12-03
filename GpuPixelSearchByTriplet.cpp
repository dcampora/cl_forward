
#include "GpuPixelSearchByTriplet.h"

int independent_execute(
    const std::vector<std::vector<uint8_t> > & input,
    std::vector<std::vector<uint8_t> > & output) {

  std::vector<const std::vector<uint8_t>* > converted_input;
  converted_input.resize(input.size());

  for (int i=0; i<input.size(); ++i) {
    converted_input[i] = &(input[i]);
  }

  std::cout << std::fixed << std::setprecision(2);
  logger::ll.verbosityLevel = 3;

  // Order input hits by X
  preorder_by_x(converted_input);

  return gpuPixelSearchByTripletInvocation(converted_input, output);
}

void independent_post_execute(const std::vector<std::vector<uint8_t> > & output) {
    // DEBUG << "post_execute invoked" << std::endl;
    DEBUG << std::endl << "Size of output: " << output.size() << " entries" << std::endl;
}

int gpuPixelSearchByTriplet(
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output) {

  // Silent execution
  std::cout << std::fixed << std::setprecision(2);
  logger::ll.verbosityLevel = 0;
  return gpuPixelSearchByTripletInvocation(input, output);
}

/**
 * Common entrypoint for Gaudi and non-Gaudi
 * @param input  
 * @param output 
 */
int gpuPixelSearchByTripletInvocation(
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output) {
  DEBUG << "Invoking gpuPixelSearchByTriplet with " << input.size() << " events" << std::endl;

  // Define how many blocks / threads we need to deal with numberOfEvents
  // Each execution will return a different output
  output.resize(input.size());
  
  // Execute one event every time (one_event branch)
  const int max_events_to_process_per_kernel = 1;

  for (int i=0; i<input.size(); i+=max_events_to_process_per_kernel){
    int events_to_process = input.size() - i;
    if (events_to_process > max_events_to_process_per_kernel)
      events_to_process = max_events_to_process_per_kernel;

    invokeParallelSearch(i, events_to_process, input, output);
  }

  return 0;
}
