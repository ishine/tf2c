#ifndef MAINUTIL_H_
#define MAINUTIL_H_

struct Tensor;

void run(int argc, const char* argv[], Tensor* (*fp)());

#endif
