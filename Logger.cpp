
#define CUDALOGGER_CPP 1

#include "Logger.h"

std::ostream& logger::logger(int requestedLogLevel){
    if (logger::ll.verbosityLevel >= requestedLogLevel){
        return std::cout;
    } else {
        return logger::ll.discardStream;
    }
}
