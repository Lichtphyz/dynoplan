#include "dynobench/motions.hpp"
#include <iostream>
#include <string>
using namespace dynobench;
int main(int argc, char *argv[]) {
  std::string in_file;
  if (argc >= 2) {
    in_file = argv[1];
  } else {
    // default: original car1_v0 file
    in_file = "../cloud/motionsV2/good/car1_v0/car1_v0_all.bin.sp.bin";
  }
  std::string out_file = (argc >= 3) ? argv[2] : in_file + ".msgpack";
  Trajectories trajs;
  trajs.load_file_boost(in_file.c_str());
  trajs.save_file_msgpack(out_file.c_str());
  std::cout << "Written to: " << out_file << std::endl;
  return 0;
}
