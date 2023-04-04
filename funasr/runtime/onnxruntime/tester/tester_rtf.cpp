
#ifndef _WIN32
#include <sys/time.h>
#else
#include <win_func.h>
#endif

#include "librapidasrapi.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

int main(int argc, char *argv[])
{

    if (argc < 4)
    {
        printf("Usage: %s /path/to/model_dir /path/to/wav.scp quantize(true or false) \n", argv[0]);
        exit(-1);
    }

    // read wav.scp
    vector<string> wav_list;
    ifstream in(argv[2]);
    if (!in.is_open()) {
        printf("Failed to open file: %s", argv[2]);
        return 0;
    }
    string line;
    while(getline(in, line))
    {
        istringstream iss(line);
        string column1, column2;
        iss >> column1 >> column2;
        wav_list.push_back(column2); 
    }
    in.close();

    // model init
    struct timeval start, end;
    gettimeofday(&start, NULL);
    int nThreadNum = 1;
    // is quantize
    bool quantize = false;
    istringstream(argv[3]) >> boolalpha >> quantize;

    RPASR_HANDLE AsrHanlde=RapidAsrInit(argv[1], nThreadNum, quantize);
    if (!AsrHanlde)
    {
        printf("Cannot load ASR Model from: %s, there must be files model.onnx and vocab.txt", argv[1]);
        exit(-1);
    }
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Model initialization takes %lfs.\n", (double)modle_init_micros / 1000000);

    // warm up
    for (size_t i = 0; i < 30; i++)
    {
        RPASR_RESULT Result=RapidAsrRecogFile(AsrHanlde, wav_list[0].c_str(), RASR_NONE, NULL, false);
    }

    // forward
    float snippet_time = 0.0f;
    float total_length = 0.0f;
    long total_time = 0;
    
    for (size_t i = 0; i < wav_list.size(); i++)
    {
        gettimeofday(&start, NULL);
        RPASR_RESULT Result=RapidAsrRecogFile(AsrHanlde, wav_list[i].c_str(), RASR_NONE, NULL);
        gettimeofday(&end, NULL);
        seconds = (end.tv_sec - start.tv_sec);
        long taking_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
        total_time += taking_micros;

        if(Result){
            string msg = RapidAsrGetResult(Result, 0);
            printf("Result: %s \n", msg.c_str());

            snippet_time = RapidAsrGetRetSnippetTime(Result);
            total_length += snippet_time;
            RapidAsrFreeResult(Result);
        }else{
            cout <<"No return data!";
        }
    }

    printf("total_time_wav %ld ms.\n", (long)(total_length * 1000));
    printf("total_time_comput %ld ms.\n", total_time / 1000);
    printf("total_rtf %05lf .\n", (double)total_time/ (total_length*1000000));

    RapidAsrUninit(AsrHanlde);
    return 0;
}
