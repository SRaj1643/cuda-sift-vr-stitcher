#include <iostream>

#include "gaussian.h"

void runGaussianTest()
{
    std::cout << "\nGaussian Test\n";

    gaussianBlur(
        nullptr,
        nullptr,
        0,
        0
    );
}
