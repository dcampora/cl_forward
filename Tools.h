
/**
 * Tools.h
 */

#ifndef TOOLS
#define TOOLS 1

#include <cstring>
#include <iostream>
#include <vector>
#include <sstream>
#include <map>
#include <cmath>
#include <stdint.h>
#include <stdexcept>

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include "Definitions.h"

#define clCheck(stmt) { \
  cl_int status = stmt; \
  if (status != CL_SUCCESS) { \
    std::cerr << "Error in function " << #stmt << std::endl; \
    return -1; \
  } \
}

template <class T>
std::string toString(T t){
    std::stringstream ss;
    std::string s;
    ss << t;
    ss >> s;
    return s;
}

int convertClToString(const char *filename, std::string& s);
void setHPointersFromInput(uint8_t * input, size_t size);
void preorder_by_x(std::vector<const std::vector<uint8_t>* > & input);

// A non-efficient implementation that does what I need
void quicksort (float* a, float* b, float* c, unsigned int* d, int start, int end);
int divide (float* a, float* b, float* c, unsigned int* d, int first, int last);
template<typename T> void swap (T& a, T& b);

std::map<std::string, float> calcResults(std::vector<float>& times);
void checkClError(const cl_int errcode_ret);

#endif
