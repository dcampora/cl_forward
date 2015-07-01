
/**
 *      FileStdLogger
 *
 *      author  -   Daniel Campora
 *      email   -   dcampora@cern.ch
 *
 *      April, 2014
 *      CERN
 */

#ifndef FILESTDLOGGER
#define FILESTDLOGGER 1

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// Dumb type, just making constructor public
class FileStdLogger : public std::ostream {
private:
    std::streambuf* b;
public:
    FileStdLogger() : std::ostream(b) {}
};

// Can't believe this is so simple
class MessageLogger : public std::streambuf {
public:
    std::string _buf;
    std::ofstream* _file_io;
    FileStdLogger* _file_std_io;

    std::streambuf* _old;

    MessageLogger(std::ofstream* file_io, FileStdLogger* file_std_io) :
        _buf(""), _file_io(file_io), _file_std_io(file_std_io) {
        // Override the previous read buffer
        _old = _file_std_io->rdbuf(this);
    }

    ~MessageLogger(){
        _file_std_io->rdbuf(_old);
    }

    int overflow(int c){
        if (c == '\n'){
            std::cout << _buf << std::endl;
            (*_file_io) << _buf << std::endl;
            _buf = "";
        }
        else
            _buf += c;

        return c;
    }
};

class VoidLogger : public std::streambuf {
public:
    std::ostream* _void_stream;
    std::streambuf* _old;

    VoidLogger(std::ostream* void_stream) : 
        _void_stream(void_stream) {
        _old = _void_stream->rdbuf(this);
    }

    ~VoidLogger(){
        _void_stream->rdbuf(_old);
    }

    int overflow(int c){
        // Just don't do anything
        return c;
    }

};

#endif
