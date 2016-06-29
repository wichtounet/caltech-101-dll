//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <vector>
#include <string>

#include "dll/conv_rbm.hpp"
#include "dll/conv_rbm_mp.hpp"
#include "dll/dbn.hpp"

namespace {

} //end of anonymous namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: caltech dataset_path" << std::endl;
        return -1;
    }

    std::vector<std::string> args;

    for (std::size_t i = 1; i < static_cast<size_t>(argc); ++i) {
        args.emplace_back(argv[i]);
    }

    std::string dataset_path = args[0];
    std::string category = "cars";

    constexpr const std::size_t width = 298;
    constexpr const std::size_t height = 199;



    return 0;
}
