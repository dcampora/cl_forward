#ifndef KERNEL_INVOKER
#define KERNEL_INVOKER 1

#include <iostream>
#include "Definitions.h"
#include "Tools.h"
#include "Logger.h"

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <map>
#include <stdint.h>
#include <assert.h>

#include <CL/cl.h>

void getMaxNumberOfHits(char*& input, int& maxHits);
void printOutSensorHits(int sensorNumber, int* prevs, int* nexts);
void printOutAllSensorHits(int* prevs, int* nexts);
void printInfo(int numberOfSensors, int numberOfHits);
void printTrack(Track* tracks, const int trackNumber, const std::map<int, int>& zhit_to_module, std::ofstream& outstream,
  TrackParameters* tp, FitKalmanTrackParameters* fktp_u, FitKalmanTrackParameters* fktp_d);
int findClosestModule(const int z, const std::map<int, int>& zhit_to_module);

int invokeParallelSearch(
    const int startingEvent,
    const int eventsToProcess,
    const std::vector<const std::vector<uint8_t>* > & input,
    std::vector<std::vector<uint8_t> > & output);

struct EventBeginning {
  int numberOfSensors;
  int numberOfHits;
};

#endif
