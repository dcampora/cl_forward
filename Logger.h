#ifndef CUDALOGGER
#define CUDALOGGER 1

#define DEBUG logger::logger(3)
#define INFO  logger::logger(2)
#define ERROR logger::logger(1)

#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "FileStdLogger.h"


namespace logger {
    class Logger {
    public:
      int verbosityLevel;
      FileStdLogger discardStream;
      VoidLogger* discardLogger;
      Logger(){
        discardLogger = new VoidLogger(&discardStream);
      }
    };

    std::ostream& logger(int requestedLogLevel);

    #ifndef CUDALOGGER_CPP
    extern Logger ll;
    #else
    Logger ll;
    #endif
}

#endif
