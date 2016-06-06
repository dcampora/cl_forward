
/**
 *      Autocontained cross-platform independent entrypoint
 *      for Gaudi offloaded algorithms.
 *
 *      Intended for testing and debugging, completely decoupled
 *      from the Gaudi framework.
 *
 *      author  -   Daniel Campora
 *      email   -   dcampora@cern.ch
 *
 *      June, 2014
 *      CERN
 */

#include <iostream>
#include <string>
#include <cstring>
#include <exception>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <dirent.h>

/**
 * execute entrypoint of algorithm
 * Same signature as offloaded gaudi-algorithm
 * 
 * @param output 
 * @param input  
 */
extern int independent_execute(
    const std::vector<std::vector<unsigned char> >& input,
    std::vector<std::vector<unsigned char> >& output);

/**
 * Post execution entrypoint
 * @param output 
 */
extern void independent_post_execute(
    const std::vector<std::vector<unsigned char> >& output);


void printUsage(char* argv[]){
    std::cerr << "Usage: "
        << argv[0] << " <folder including .dat files> [number of files to process=100]"
        << std::endl;
}

/**
 * Generic StrException launcher
 */
class StrException : public std::exception
{
public:
    std::string s;
    StrException(std::string ss) : s(ss) {}
    ~StrException() throw () {} // Updated
    const char* what() const throw() { return s.c_str(); }
};

/**
 * Checks file existence
 * @param  name 
 * @return      
 */
bool fileExists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

/**
 * Reads some data from an input file, following the
 * specified format of choice
 *
 * Format expected in file:
 *
 * int funcNameLen
 * char* funcName
 * int dataSize
 * char* data
 */
void readFileIntoVector(std::string filename, std::vector<unsigned char>& output){
    // Check if file exists
    if (!fileExists(filename)){
        throw StrException("Error: File " + filename + " does not exist.");
    }

    std::ifstream infile (filename.c_str(), std::ifstream::binary);

    // get size of file
    infile.seekg(0, std::ifstream::end);
    int size = infile.tellg();
    infile.seekg(0);

    // Read format expected:
    //  int funcNameLen
    //  char* funcName
    //  int dataSize
    //  char* data
    int funcNameLen;
    int dataSize;
    std::vector<char> funcName;

    char* pFuncNameLen = (char*) &funcNameLen;
    char* pDataSize = (char*) &dataSize;
    infile.read(pFuncNameLen, sizeof(int));

    funcName.resize(funcNameLen);
    infile.read(&(funcName[0]), funcNameLen);
    infile.read(pDataSize, sizeof(int));

    // read content of infile with a vector
    output.resize(dataSize);
    infile.read ((char*) &(output[0]), dataSize);
    infile.close();
}

/**
 * This is if the function is called on its own
 * (ie. non-gaudi execution)
 * 
 * In that case, the file input is expected.
 * As a convention, multiple files would be specified
 * with comma-separated values
 * 
 * @param  argc
 * @param  argv
 * @return     
 */
int main(int argc, char *argv[])
{
    std::vector<std::vector<unsigned char>> input;
    std::string folderName;
    int fileNumber = 1;
    size_t numberOfFilesToProcess = 50;

    // Get params (getopt independent)
    if (argc < 2){
        printUsage(argv);
        return -1;
    }
    
    folderName = std::string(argv[1]);
    if (argc >= 3) {
        numberOfFilesToProcess = atoi(argv[2]);
    }

    input.resize(numberOfFilesToProcess);
    std::vector<std::string> dataFilesInFolder;

    // Directory listing in C
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (folderName.c_str())) != nullptr) {
        while ((ent = readdir (dir)) != nullptr) {
            // std::string fn = "filename.conf";
            // if(fn.substr(fn.find_last_of(".") + 1) == "conf") {
            std::string filename = std::string(ent->d_name);
            if (filename.substr(filename.find_last_of(".") + 1) == "dat") {
                dataFilesInFolder.push_back(filename);
            }
        }
        closedir (dir);
    }
    if (dir == nullptr || dataFilesInFolder.empty()) {
        std::cerr << "Directory " << folderName << " either could not be opened, or is empty" << std::endl;
        return -1;
    }

    // Process in round robin the files required
    for (unsigned int i=0; i<numberOfFilesToProcess; ++i) {
        unsigned int filenumber = i % dataFilesInFolder.size();
        readFileIntoVector(folderName + "/" + dataFilesInFolder[filenumber], input[i]);
    }

    // Print out first byte from formatter->inputPointer
    std::cout << input.size() << " files read" << std::endl;

    // Call offloaded algo
    std::vector<std::vector<unsigned char>> output;
    independent_execute(input, output);

    // Post execution entrypoint
    independent_post_execute(output);

    return 0;
}
