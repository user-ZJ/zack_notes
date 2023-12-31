#include "utils/flags.h"
#include "utils/logging.h"
#include <cereal/archives/json.hpp>
#include <cereal/archives/optional_json.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
#include <optional>


#define MESSAGE_DESERIALIZE                                                    \
  int deserialize(const std::string &str) {                                   \
    try {                                                                      \
      std::stringstream ss(str);                                               \
      cereal::JSONInputArchive archive(ss);                                    \
      serialize(archive);                                                      \
      return 0;                                                                \
    } catch (std::exception & e) {                                             \
      LOG(ERROR) << e.what();                                                  \
      return 1;                                                                \
    }                                                                          \
  }

#define MESSAGE_SERIALIZE                                                      \
  std::string serialize() {                                                    \
    std::stringstream ss;                                                      \
    try {                                                                      \
      cereal::JSONOutputArchive archive(                                       \
          ss, cereal::JSONOutputArchive::Options::NoIndent());                 \
      serialize(archive);                                                      \
    } catch (std::exception & e) {                                             \
      LOG(ERROR) << e.what();                                                  \
    }                                                                          \
    return ss.str();                                                           \
  }

struct MyData {
  int x, y, z;
  std::string word = "你好";

  std::string serialize() {
    std::stringstream ss;
    try {
      {
        cereal::JSONOutputArchive archive(ss);
        save(archive);
      }
    } catch (std::exception &e) {
      LOG(ERROR) << e.what();
    }
    return ss.str();
  }

  void deserialize(const std::string &str) {
    try {
      std::stringstream ss(str);
      cereal::JSONInputArchive archive(ss);
      load(archive);
    } catch (std::exception &e) {
      LOG(ERROR) << e.what();
    }
  }

//   template <class Archive> void save(Archive &ar) const {
//     ar(CEREAL_NVP(x), CEREAL_NVP(y), CEREAL_NVP(z), CEREAL_NVP(word));
//   }

//   template <class Archive> void load(Archive &ar) {
//     ar(cereal::make_optional_nvp("x", x));
//     ar(cereal::make_optional_nvp("z", z));
//     ar(CEREAL_NVP(y));
//   }

  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_optional_nvp("x", x));
    ar(cereal::make_optional_nvp("z", z));
    ar(CEREAL_NVP(y));
  }

  MyData() : x(0), y(0), z(0) {}
};

int main(int argc, char *argv[]) {
  google::EnableLogCleaner(30); // keep your logs for 30 days
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]); //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  google::InstallFailureSignalHandler();
  std::stringstream ss;
  MyData myData;
  {
    cereal::JSONOutputArchive outarchive(std::cout);
    myData.save(outarchive);
  }

  std::string sjson = R"({"y": 2,"z": 3,"i":8})";
  ss << sjson;
  {
    cereal::JSONInputArchive inarchive(ss);
    myData.load(inarchive);
  }
  LOG(INFO) << myData.x << " " << myData.y << " " << myData.z;

  return 0;
}