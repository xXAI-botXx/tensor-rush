#pragma once


#ifdef TENSOR_RUSH_EXPORTS
#define TENSOR_RUSH_API __declspec(dllexport)
#else
#define TENSOR_RUSH_API __declspec(dllimport)
#endif

// Submodules
#include "neural-network.h"
#include "math.h"

namespace rush {


    class TENSOR_RUSH_API MeineKlasse {
    public:
        MeineKlasse();
        void sagHallo();
    };


}


