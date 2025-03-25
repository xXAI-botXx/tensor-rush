#pragma once


#ifdef TENSOR_RUSH_EXPORTS
#define TENSOR_RUSH_API __declspec(dllexport)
#else
#define TENSOR_RUSH_API __declspec(dllimport)
#endif


class TENSOR_RUSH_API MeineKlasse {
public:
    MeineKlasse();
    void sagHallo();
};



