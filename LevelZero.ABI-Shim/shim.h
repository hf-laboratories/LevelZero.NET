#ifndef LevelZero_SHIM_H
#define LevelZero_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

// Level Zero C ABI shim (minimal entrypoints)

#include <stdint.h>

#if defined(_WIN32)
	#define LZ_API __declspec(dllexport)
#elif defined(__GNUC__)
	#define LZ_API __attribute__((visibility("default")))
#else
	#define LZ_API
#endif

// Initialize Level Zero (zeInit)
LZ_API int32_t lz_init(uint32_t flags);

// Returns the number of drivers available
LZ_API int32_t lz_get_driver_count(uint32_t* count);

// Returns a driver handle for a given index
LZ_API int32_t lz_get_driver_handle(uint32_t driver_index, void** driver);

// Returns the number of devices for a driver index
LZ_API int32_t lz_get_device_count(uint32_t driver_index, uint32_t* count);

// Writes a device name into caller-provided buffer
LZ_API int32_t lz_get_device_name(uint32_t driver_index, uint32_t device_index, char* buffer, uint32_t buffer_size);

// Returns a device handle for a given driver/device index
LZ_API int32_t lz_get_device_handle(uint32_t driver_index, uint32_t device_index, void** device);

// Context management
LZ_API int32_t lz_context_create(void* driver, void** context);
LZ_API int32_t lz_context_destroy(void* context);

// Command queue/list
LZ_API int32_t lz_command_queue_create(void* context, void* device, void** queue);
LZ_API int32_t lz_command_queue_destroy(void* queue);
LZ_API int32_t lz_command_list_create(void* context, void* device, void** command_list);
LZ_API int32_t lz_command_list_reset(void* command_list);
LZ_API int32_t lz_command_list_close(void* command_list);
LZ_API int32_t lz_command_list_destroy(void* command_list);
LZ_API int32_t lz_command_queue_execute(void* queue, void* command_list);
LZ_API int32_t lz_command_queue_synchronize(void* queue, uint64_t timeout_ns);

// Module / kernel
LZ_API int32_t lz_module_create(void* context, void* device, const uint8_t* module_bytes, uint32_t size, uint32_t module_format, void** module, char* build_log, uint32_t build_log_size);
LZ_API int32_t lz_module_destroy(void* module);
LZ_API int32_t lz_kernel_create(void* module, const char* name, void** kernel);
LZ_API int32_t lz_kernel_destroy(void* kernel);
LZ_API int32_t lz_kernel_set_group_size(void* kernel, uint32_t group_x, uint32_t group_y, uint32_t group_z);
LZ_API int32_t lz_kernel_set_arg_value(void* kernel, uint32_t index, size_t size, const void* value);
LZ_API int32_t lz_kernel_set_arg_mem(void* kernel, uint32_t index, const void* ptr);

// Command list operations
LZ_API int32_t lz_command_list_append_launch_kernel(void* command_list, void* kernel, uint32_t group_x, uint32_t group_y, uint32_t group_z);
LZ_API int32_t lz_command_list_append_barrier(void* command_list);
LZ_API int32_t lz_command_list_append_memory_copy(void* command_list, void* dst, const void* src, size_t size);

// Unified Shared Memory
LZ_API int32_t lz_usm_alloc_shared(void* context, void* device, size_t size, size_t alignment, void** ptr);
LZ_API int32_t lz_usm_free(void* context, void* ptr);

// Error helpers
LZ_API const char* lz_get_last_error();
LZ_API int32_t lz_get_last_result();
LZ_API void lz_clear_error();

#ifdef __cplusplus
}
#endif

#endif // LevelZero_SHIM_H
