/*******************************************************************************
 * Copyright 上海赜睿信息科技有限公司(Zilliz) - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 ******************************************************************************/
#pragma once

#include <string>
#include <time.h>

#include "Error.h"

namespace zilliz {
namespace milvus {
namespace server {

class CommonUtil {
 public:
    static bool GetSystemMemInfo(unsigned long &total_mem, unsigned long &free_mem);
    static bool GetSystemAvailableThreads(unsigned int &thread_count);

    static bool IsFileExist(const std::string &path);
    static uint64_t GetFileSize(const std::string &path);
    static bool IsDirectoryExist(const std::string &path);
    static ServerError CreateDirectory(const std::string &path);
    static ServerError DeleteDirectory(const std::string &path);

    static std::string GetExePath();

    static bool TimeStrToTime(const std::string& time_str,
            time_t &time_integer,
            tm &time_struct,
            const std::string& format = "%d-%d-%d %d:%d:%d");

    static void ConvertTime(time_t time_integer, tm &time_struct);
    static void ConvertTime(tm time_struct, time_t &time_integer);
};

}
}
}

