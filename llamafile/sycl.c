// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2023 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llama.cpp/ggml-backend-impl.h"
#include "llama.cpp/ggml-sycl.h"
#include "llamafile/llamafile.h"
#include "llamafile/log.h"
#include "llamafile/x.h"
#include <assert.h>
#include <cosmo.h>
#include <dlfcn.h>
#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

__static_yoink("llama.cpp/ggml.h");
__static_yoink("llamafile/compcap.cu");
__static_yoink("llama.cpp/ggml-impl.h");
__static_yoink("llamafile/llamafile.h");
__static_yoink("llama.cpp/ggml-sycl.cpp");
__static_yoink("llama.cpp/ggml-sycl.h");
__static_yoink("llama.cpp/ggml-alloc.h");
__static_yoink("llama.cpp/ggml-common.h");
__static_yoink("llama.cpp/ggml-backend.h");
__static_yoink("llama.cpp/ggml-backend-impl.h");

// TODO CLEAN UP LIB LINK
#define SYCL_FLAGS "-fsycl", "--shared", "-fPIC", "-DGGML_SYCL_F16", \
        "-DNDEBUG", "-DGGML_BUILD=1", "-DGGML_SHARED=1", "-DGGML_MULTIPLATFORM", \
        "-march=native",  "-mtune=native", \
        "-Wno-deprecated-declarations", "-Wno-write-strings", \
        "-Wno-switch", "-L${MKLROOT}/lib", "-g", "-O0"

#define SYCL_LIBS "-lOpenCL", "-lmkl_core", "-lpthread", "-lmkl_sycl_blas", \
        "-lmkl_intel_ilp64", "-lmkl_tbb_thread"
// TODO WINDOWS IMPORTS
// #define SYCL_LIBS "-lOpenCL", "-lsycl7", "-lmkl_sycl_blas_dll.lib", "-lmkl_intel_ilp64_dll.lib", "-lmkl_sequential_dll.lib", "-lmkl_core_dll.lib"

static const struct Source {
    const char *zip;
    const char *name;
} srcs[] = {
    {"/zip/llama.cpp/ggml.h", "ggml.h"},
    {"/zip/llamafile/llamafile.h", "llamafile.h"},
    {"/zip/llama.cpp/ggml-impl.h", "ggml-impl.h"},
    {"/zip/llama.cpp/ggml-sycl.h", "ggml-sycl.h"},
    {"/zip/llama.cpp/ggml-alloc.h", "ggml-alloc.h"},
    {"/zip/llama.cpp/ggml-common.h", "ggml-common.h"},
    {"/zip/llama.cpp/ggml-backend.h", "ggml-backend.h"},
    {"/zip/llama.cpp/ggml-backend-impl.h", "ggml-backend-impl.h"},
    {"/zip/llama.cpp/ggml-sycl.cpp", "ggml-sycl.cpp"} // should be last
};


GGML_CALL int ggml_backend_sycl_reg_devices(void);

static struct Sycl {
    bool supported;
    atomic_uint once;
    typeof(ggml_backend_sycl_reg_devices) *GGML_CALL reg_devices;

    typeof(ggml_sycl_link) *GGML_CALL link;
    typeof(ggml_backend_sycl_init) *GGML_CALL backend_init;
    typeof(ggml_backend_sycl_buffer_type) *GGML_CALL buffer_type;
    typeof(ggml_backend_sycl_host_buffer_type) *GGML_CALL host_buffer_type;
    typeof(ggml_backend_sycl_split_buffer_type) *GGML_CALL split_buffer_type;

    typeof(ggml_backend_sycl_print_sycl_devices) *print_sycl_devices;
    typeof(ggml_sycl_get_gpu_list) *GGML_CALL get_gpu_list;
    typeof(ggml_sycl_get_device_description) *GGML_CALL get_device_description;
    typeof(ggml_backend_sycl_get_device_count) *GGML_CALL get_device_count;
    typeof(ggml_backend_sycl_get_device_memory) *GGML_CALL get_device_memory;
    typeof(ggml_backend_sycl_get_device_index) *GGML_CALL get_device_index;

    typeof(ggml_backend_sycl_get_device_id) *GGML_CALL get_device_id;
    typeof(ggml_backend_sycl_set_single_device_mode) *GGML_CALL set_single_device_mode;
    typeof(ggml_backend_sycl_set_mul_device_mode) *GGML_CALL set_mul_device_mode;
} ggml_sycl;

static const char *Dlerror(void) {
    const char *msg;
    msg = cosmo_dlerror();
    if (!msg)
        msg = "null dlopen error";
    return msg;
}

static const char *GetDsoExtension(void) {
    if (IsWindows())
        return "dll";
    else if (IsXnu())
        return "dylib";
    else
        return "so";
}

static bool FileExists(const char *path) {
    struct stat st;
    return !stat(path, &st);
}

static bool IsExecutable(const char *path) {
    struct stat st;
    return !stat(path, &st) && (st.st_mode & 0111) && !S_ISDIR(st.st_mode);
}

static bool CreateTempPath(const char *path, char tmp[static PATH_MAX]) {
    int fd;
    strlcpy(tmp, path, PATH_MAX);
    strlcat(tmp, ".XXXXXX", PATH_MAX);
    if ((fd = mkostemp(tmp, O_CLOEXEC)) != -1) {
        close(fd);
        return true;
    } else {
        perror(tmp);
        return false;
    }
}

static bool Compile(const char *src, const char *tmp, const char *out, char *args[]) {
    int pid, ws;
    llamafile_log_command(args);
    errno_t err = posix_spawnp(&pid, args[0], NULL, NULL, args, environ);
    if (err) {
        perror(args[0]);
        unlink(tmp);
        return false;
    }
    while (waitpid(pid, &ws, 0) == -1) {
        if (errno != EINTR) {
            perror(args[0]);
            unlink(tmp);
            return false;
        }
    }
    if (ws) {
        tinylog(__func__, ": warning: ", args[0], " returned nonzero exit status\n", NULL);
        unlink(tmp);
        return false;
    }
    if (rename(tmp, out)) {
        perror(out);
        unlink(tmp);
        return false;
    }
    return true;
}

// finds sycl compiler
//
//   1. icpx on $PATH environ
//   2. $COMPLR_ROOT/bin/icpx
//   3. /opt/intel/oneapi/compiler/latest/bin/icpx
//

static bool get_compiler_path(char path[static PATH_MAX]) {
    const char *name = IsWindows() ? "icpx.exe" : "icpx";
    if (commandv(name, path, PATH_MAX))
        return true;
    else
        tinylog(__func__, ": note: ", name, " not found on $PATH\n", NULL);
    const char *compiler_path;
    if ((compiler_path = getenv("CMPLR_ROOT"))) {
        if (!*compiler_path)
            return false;
        strlcpy(path, compiler_path, PATH_MAX);
        strlcat(path, "/bin/", PATH_MAX);
    } else {
        tinylog(__func__, ": note: $CMPLR_ROOT/bin/", name, " does not exist\n", NULL);
        strlcpy(path, "/opt/intel/oneapi/compiler/latest/bin/", PATH_MAX);
    }
    strlcat(path, name, PATH_MAX);
    if (IsExecutable(path)) {
        return true;
    } else {
        tinylog(__func__, ": note: ", path, " does not exist\n", NULL);
        return false;
    }
}

static bool compile_sycl(const char *compiler, const char *dso, const char *src) {

    // create temporary output path for atomicity
    char tmpdso[PATH_MAX];
    if (!CreateTempPath(dso, tmpdso))
        return false;

    // try building dso with host SYCL
    tinylog(__func__, ": note: building ggml-sycl...\n", NULL);
    if (Compile(src, tmpdso, dso,
                (char *[]){(char *)compiler, SYCL_FLAGS, "-o", tmpdso, (char *)src,
                           SYCL_LIBS, NULL}))
        return true;

    // oh no
    return false;
}

static bool extract_sycl_dso(const char *dso, const char *name) {

    // see if prebuilt dso is bundled in zip assets
    char zip[80];
    strlcpy(zip, "/zip/", sizeof(zip));
    strlcat(zip, name, sizeof(zip));
    strlcat(zip, ".", sizeof(zip));
    strlcat(zip, GetDsoExtension(), sizeof(zip));
    if (!FileExists(zip)) {
        tinylog(__func__, ": note: prebuilt binary ", zip, " not found\n", NULL);
        return false;
    }

    // extract prebuilt dso
    return llamafile_extract(zip, dso);
}

static void *imp(void *lib, const char *sym) {
    void *fun = cosmo_dlsym(lib, sym);
    if (!fun)
        tinylog(__func__, ": error: failed to import symbol: ", sym, "\n", NULL);
    return fun;
}

static bool link_sycl_dso(const char *dso) {
    // runtime link dynamic shared object
    void *lib;
    tinylog(__func__, ": note: dynamically linking ", dso, "\n", NULL);

    lib = cosmo_dlopen(dso, RTLD_LAZY);

    if (!lib) {
        char cc[PATH_MAX];
        tinylog(__func__, ": warning: ", Dlerror(), ": failed to load library\n", NULL);
        if ((IsLinux() || IsBsd()) && !commandv("icpx", cc, PATH_MAX))
            tinylog(__func__, ": note: you need to install icpx for sycl gpu support\n", NULL);
        return false;
    }

    // import functions
    bool ok = true;
    ok &= !!(ggml_sycl.reg_devices = imp(lib, "ggml_backend_sycl_reg_devices"));
    ok &= !!(ggml_sycl.link = imp(lib, "ggml_sycl_link"));
    ok &= !!(ggml_sycl.backend_init = imp(lib, "ggml_backend_sycl_init"));
    ok &= !!(ggml_sycl.buffer_type = imp(lib, "ggml_backend_sycl_buffer_type"));
    ok &= !!(ggml_sycl.host_buffer_type = imp(lib, "ggml_backend_sycl_host_buffer_type"));
    ok &= !!(ggml_sycl.split_buffer_type = imp(lib, "ggml_backend_sycl_split_buffer_type"));

    ok &= !!(ggml_sycl.print_sycl_devices = imp(lib, "ggml_backend_sycl_print_sycl_devices"));
    ok &= !!(ggml_sycl.get_gpu_list = imp(lib, "ggml_sycl_get_gpu_list"));
    ok &= !!(ggml_sycl.get_device_description = imp(lib, "ggml_sycl_get_device_description"));
    ok &= !!(ggml_sycl.get_device_count = imp(lib, "ggml_backend_sycl_get_device_count"));
    ok &= !!(ggml_sycl.get_device_memory = imp(lib, "ggml_backend_sycl_get_device_memory"));
    ok &= !!(ggml_sycl.get_device_index = imp(lib, "ggml_backend_sycl_get_device_index"));

    ok &= !!(ggml_sycl.get_device_id = imp(lib, "ggml_backend_sycl_get_device_id"));
    ok &= !!(ggml_sycl.set_single_device_mode = imp(lib, "ggml_backend_sycl_set_single_device_mode"));
    ok &= !!(ggml_sycl.set_mul_device_mode = imp(lib, "ggml_backend_sycl_set_mul_device_mode"));

    if (!ok) {
        tinylog(__func__, ": error: not all sycl symbols could be imported\n", NULL);
        cosmo_dlclose(lib);
        return false;
    }

    // ask the library if actual gpu devices exist
    if (ggml_sycl.link(ggml_backend_api())) {
        tinylog(__func__, ": GPU support loaded\n", NULL);
        return true;
    } else {
        tinylog(__func__, ": No GPU devices found\n", NULL);
        cosmo_dlclose(lib);
        return false;
    }
    return true;
}

static bool import_sycl_impl(void) {

    // No dynamic linking support on OpenBSD yet.
    if (IsOpenbsd())
        return false;

    // Check if we're allowed to even try.
    switch (FLAG_gpu) {
    case LLAMAFILE_GPU_AUTO:
    case LLAMAFILE_GPU_SYCL:
        break;
    default:
        return false;
    }
    tinylog(__func__, ": initializing gpu module...\n", NULL);

    // extract source code
    char src[PATH_MAX];
    bool needs_rebuild = FLAG_recompile;
    for (int i = 0; i < sizeof(srcs) / sizeof(*srcs); ++i) {
        llamafile_get_app_dir(src, sizeof(src));
        if (!i && mkdir(src, 0755) && errno != EEXIST) {
            perror(src);
            return false;
        }
        strlcat(src, srcs[i].name, sizeof(src));
        switch (llamafile_is_file_newer_than(srcs[i].zip, src)) {
        case -1:
            return false;
        case false:
            break;
        case true:
            needs_rebuild = true;
            if (!llamafile_extract(srcs[i].zip, src))
                return false;
            break;
        default:
            __builtin_unreachable();
        }
    }

    char dso[PATH_MAX];
    char bindir[PATH_MAX];
    const char *compiler_path;
    char compiler_path_buf[PATH_MAX];

    // Attempt to load SYCL GPU support.
    switch (FLAG_gpu) {
    case LLAMAFILE_GPU_AUTO:
    case LLAMAFILE_GPU_SYCL:
        // Get some essential paths.
        // Assume compiler and library path are the same
        if (get_compiler_path(compiler_path_buf)) {
            compiler_path = compiler_path_buf;
        } else {
            compiler_path = 0;
        }

        // Get path of GGML DSO for SYCL.
        llamafile_get_app_dir(dso, PATH_MAX);
        strlcat(dso, "ggml-sycl.", PATH_MAX);
        strlcat(dso, GetDsoExtension(), PATH_MAX);
        if (FLAG_nocompile)
            return ((FileExists(dso) || extract_sycl_dso(dso, "ggml-sycl")) &&
                    link_sycl_dso(dso));

        // Check if DSO is already compiled.
        if (!needs_rebuild && !FLAG_recompile) {
            switch (llamafile_is_file_newer_than(src, dso)) {
            case -1:
                return false;
            case false:
                return link_sycl_dso(dso);
            case true:
                break;
            default:
                __builtin_unreachable();
            }
        }

        // Try building SYCL from source
        if (compiler_path && compile_sycl(compiler_path, dso, src))
            return link_sycl_dso(dso);

        break;
    default:
        break;
    }

    // too bad
    return false;
}

static void import_sycl(void) {
    if (llamafile_has_metal())
        return;
    if (import_sycl_impl()) {
        ggml_sycl.supported = true;
    } else if (FLAG_gpu == LLAMAFILE_GPU_SYCL) {
        tinyprint(2, "fatal error: support for --gpu ", llamafile_describe_gpu(), 
                  " was explicitly requested, but it wasn't available\n", NULL);
        exit(1);
    }
}

bool llamafile_has_sycl(void) {
    cosmo_once(&ggml_sycl.once, import_sycl);
    return ggml_sycl.supported;
}

GGML_API int ggml_backend_sycl_reg_devices() {
    if (!llamafile_has_sycl())
        return 0;
    return ggml_sycl.reg_devices();
}

GGML_API ggml_backend_t ggml_backend_sycl_init(int device) {
    if (!llamafile_has_sycl())
        return 0;
    return ggml_sycl.backend_init(device);
}

// devide buffer
GGML_API ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device) {
    if (!llamafile_has_sycl())
        return 0;
    return ggml_sycl.buffer_type(device);
}

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split) {
    if (!llamafile_has_sycl())
        return 0;
    return ggml_sycl.split_buffer_type(tensor_split);
}

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type(void) {
    if (!llamafile_has_sycl())
        return 0;
    return ggml_sycl.host_buffer_type();
}

GGML_API void ggml_backend_sycl_print_sycl_devices(void) {
    if (!llamafile_has_sycl())
        return;
    ggml_sycl.print_sycl_devices();
}

GGML_API GGML_CALL void ggml_sycl_get_gpu_list(int *id_list, int max_len) {
    if (!llamafile_has_sycl())
        return;
    ggml_sycl.get_gpu_list(id_list, max_len);
}

GGML_API GGML_CALL void ggml_sycl_get_device_description(int device, char *description, size_t description_size) {
    if (!llamafile_has_sycl())
        return;
    ggml_sycl.get_device_description(device, description, description_size);
}

GGML_API GGML_CALL int ggml_backend_sycl_get_device_count() {
    if (!llamafile_has_sycl())
        return 0;
    return ggml_sycl.get_device_count();
}
GGML_API GGML_CALL void ggml_backend_sycl_get_device_memory(int device, size_t *free, size_t *total) {
    if (!llamafile_has_sycl())
        return;
    ggml_sycl.get_device_memory(device, free, total);
}
GGML_API GGML_CALL int ggml_backend_sycl_get_device_index(int device_id) {
    if (!llamafile_has_sycl())
        return 0;
    return ggml_sycl.get_device_index(device_id);
}

// TODO: these are temporary
//       ref: https://github.com/ggerganov/llama.cpp/pull/6022#issuecomment-1992615670
GGML_API GGML_CALL int ggml_backend_sycl_get_device_id(int device_index) {
    if (!llamafile_has_sycl())
        return 0;
    return ggml_sycl.get_device_id(device_index);
}

GGML_API GGML_CALL void ggml_backend_sycl_set_single_device_mode(int main_gpu_id) {
    if (!llamafile_has_sycl())
        return;
    ggml_sycl.set_single_device_mode(main_gpu_id);
}
GGML_API GGML_CALL void ggml_backend_sycl_set_mul_device_mode() {
    if (!llamafile_has_sycl())
        return;
    ggml_sycl.set_mul_device_mode();
}
