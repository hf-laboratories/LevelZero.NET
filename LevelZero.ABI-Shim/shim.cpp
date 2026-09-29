#include "shim.h"

#include <cstdio>
#include <string>
#include <vector>

#include "ze_api.h"

namespace {
	thread_local std::string g_last_error;
	thread_local int32_t g_last_result = 0;
	bool g_initialized = false;

	void set_last_error(ze_result_t result, const char* message) {
		g_last_result = static_cast<int32_t>(result);
		g_last_error = message ? message : "";
	}

	ze_result_t ensure_init() {
		if (g_initialized) {
			return ZE_RESULT_SUCCESS;
		}

		ze_result_t result = zeInit(0);
		if (result != ZE_RESULT_SUCCESS) {
			set_last_error(result, "zeInit failed");
			return result;
		}

		g_initialized = true;
		return ZE_RESULT_SUCCESS;
	}

	ze_result_t get_drivers(std::vector<ze_driver_handle_t>& drivers) {
		ze_result_t result = ensure_init();
		if (result != ZE_RESULT_SUCCESS) {
			return result;
		}

		uint32_t count = 0;
		result = zeDriverGet(&count, nullptr);
		if (result != ZE_RESULT_SUCCESS) {
			set_last_error(result, "zeDriverGet (count) failed");
			return result;
		}

		drivers.resize(count);
		result = zeDriverGet(&count, drivers.data());
		if (result != ZE_RESULT_SUCCESS) {
			set_last_error(result, "zeDriverGet (handles) failed");
			drivers.clear();
			return result;
		}

		return ZE_RESULT_SUCCESS;
	}

	bool try_get_module_format(uint32_t module_format, ze_module_format_t& resolved_format) {
		switch (module_format) {
		case 0:
			resolved_format = ZE_MODULE_FORMAT_IL_SPIRV;
			return true;
		case 1:
			resolved_format = ZE_MODULE_FORMAT_NATIVE;
			return true;
		default:
			set_last_error(ZE_RESULT_ERROR_INVALID_ARGUMENT, "module format is unsupported");
			return false;
		}
	}
}

extern "C" {

int32_t lz_init(uint32_t flags) {
#if defined(L0_STUB)
	(void)flags;
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	ze_result_t result = zeInit(flags);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeInit failed");
	} else {
		g_initialized = true;
		g_last_result = static_cast<int32_t>(result);
		g_last_error.clear();
	}
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_get_driver_count(uint32_t* count) {
	if (!count) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "count pointer is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

#if defined(L0_STUB)
	*count = 0;
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	std::vector<ze_driver_handle_t> drivers;
	ze_result_t result = get_drivers(drivers);
	if (result != ZE_RESULT_SUCCESS) {
		*count = 0;
		return static_cast<int32_t>(result);
	}

	*count = static_cast<uint32_t>(drivers.size());
	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_get_driver_handle(uint32_t driver_index, void** driver) {
	if (!driver) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "driver pointer is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

#if defined(L0_STUB)
	*driver = nullptr;
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	std::vector<ze_driver_handle_t> drivers;
	ze_result_t result = get_drivers(drivers);
	if (result != ZE_RESULT_SUCCESS) {
		*driver = nullptr;
		return static_cast<int32_t>(result);
	}

	if (driver_index >= drivers.size()) {
		set_last_error(ZE_RESULT_ERROR_INVALID_ARGUMENT, "driver index out of range");
		*driver = nullptr;
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT);
	}

	*driver = drivers[driver_index];
	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_get_device_count(uint32_t driver_index, uint32_t* count) {
	if (!count) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "count pointer is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

#if defined(L0_STUB)
	*count = 0;
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	std::vector<ze_driver_handle_t> drivers;
	ze_result_t result = get_drivers(drivers);
	if (result != ZE_RESULT_SUCCESS) {
		*count = 0;
		return static_cast<int32_t>(result);
	}

	if (driver_index >= drivers.size()) {
		set_last_error(ZE_RESULT_ERROR_INVALID_ARGUMENT, "driver index out of range");
		*count = 0;
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT);
	}

	uint32_t deviceCount = 0;
	result = zeDeviceGet(drivers[driver_index], &deviceCount, nullptr);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeDeviceGet (count) failed");
		*count = 0;
		return static_cast<int32_t>(result);
	}

	*count = deviceCount;
	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_get_device_name(uint32_t driver_index, uint32_t device_index, char* buffer, uint32_t buffer_size) {
	if (!buffer || buffer_size == 0) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "buffer is null or empty");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

#if defined(L0_STUB)
	buffer[0] = '\0';
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	std::vector<ze_driver_handle_t> drivers;
	ze_result_t result = get_drivers(drivers);
	if (result != ZE_RESULT_SUCCESS) {
		buffer[0] = '\0';
		return static_cast<int32_t>(result);
	}

	if (driver_index >= drivers.size()) {
		set_last_error(ZE_RESULT_ERROR_INVALID_ARGUMENT, "driver index out of range");
		buffer[0] = '\0';
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT);
	}

	uint32_t deviceCount = 0;
	result = zeDeviceGet(drivers[driver_index], &deviceCount, nullptr);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeDeviceGet (count) failed");
		buffer[0] = '\0';
		return static_cast<int32_t>(result);
	}

	if (device_index >= deviceCount) {
		set_last_error(ZE_RESULT_ERROR_INVALID_ARGUMENT, "device index out of range");
		buffer[0] = '\0';
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT);
	}

	std::vector<ze_device_handle_t> devices(deviceCount);
	result = zeDeviceGet(drivers[driver_index], &deviceCount, devices.data());
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeDeviceGet (handles) failed");
		buffer[0] = '\0';
		return static_cast<int32_t>(result);
	}

	ze_device_properties_t props = {};
	props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
	props.pNext = nullptr;
	result = zeDeviceGetProperties(devices[device_index], &props);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeDeviceGetProperties failed");
		buffer[0] = '\0';
		return static_cast<int32_t>(result);
	}

	std::snprintf(buffer, buffer_size, "%s", props.name);
	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_get_device_handle(uint32_t driver_index, uint32_t device_index, void** device) {
	if (!device) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "device pointer is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

#if defined(L0_STUB)
	*device = nullptr;
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	std::vector<ze_driver_handle_t> drivers;
	ze_result_t result = get_drivers(drivers);
	if (result != ZE_RESULT_SUCCESS) {
		*device = nullptr;
		return static_cast<int32_t>(result);
	}

	if (driver_index >= drivers.size()) {
		set_last_error(ZE_RESULT_ERROR_INVALID_ARGUMENT, "driver index out of range");
		*device = nullptr;
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT);
	}

	uint32_t count = 0;
	result = zeDeviceGet(drivers[driver_index], &count, nullptr);
	if (result != ZE_RESULT_SUCCESS || count == 0) {
		set_last_error(result, "zeDeviceGet (count) failed");
		*device = nullptr;
		return static_cast<int32_t>(result);
	}

	std::vector<ze_device_handle_t> devices(count);
	result = zeDeviceGet(drivers[driver_index], &count, devices.data());
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeDeviceGet (handles) failed");
		*device = nullptr;
		return static_cast<int32_t>(result);
	}

	if (device_index >= devices.size()) {
		set_last_error(ZE_RESULT_ERROR_INVALID_ARGUMENT, "device index out of range");
		*device = nullptr;
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT);
	}

	*device = devices[device_index];
	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_context_create(void* driver, void** context) {
	if (!context) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "context pointer is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

#if defined(L0_STUB)
	*context = nullptr;
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!driver) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "driver handle is null");
		*context = nullptr;
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_context_desc_t desc = {};
	desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
	desc.pNext = nullptr;

	ze_context_handle_t ctx = nullptr;
	ze_result_t result = zeContextCreate(reinterpret_cast<ze_driver_handle_t>(driver), &desc, &ctx);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeContextCreate failed");
		*context = nullptr;
		return static_cast<int32_t>(result);
	}

	*context = ctx;
	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_context_destroy(void* context) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!context) {
		return static_cast<int32_t>(ZE_RESULT_SUCCESS);
	}

	ze_result_t result = zeContextDestroy(reinterpret_cast<ze_context_handle_t>(context));
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeContextDestroy failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_command_queue_create(void* context, void* device, void** queue) {
	if (!queue) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "queue pointer is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

#if defined(L0_STUB)
	*queue = nullptr;
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!context || !device) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "context/device handle is null");
		*queue = nullptr;
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_command_queue_desc_t desc = {};
	desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
	desc.pNext = nullptr;
	desc.ordinal = 0;
	desc.index = 0;
	desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
	desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

	ze_command_queue_handle_t handle = nullptr;
	ze_result_t result = zeCommandQueueCreate(
		reinterpret_cast<ze_context_handle_t>(context),
		reinterpret_cast<ze_device_handle_t>(device),
		&desc,
		&handle);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeCommandQueueCreate failed");
		*queue = nullptr;
		return static_cast<int32_t>(result);
	}

	*queue = handle;
	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_command_queue_destroy(void* queue) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!queue) {
		return static_cast<int32_t>(ZE_RESULT_SUCCESS);
	}

	ze_result_t result = zeCommandQueueDestroy(reinterpret_cast<ze_command_queue_handle_t>(queue));
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeCommandQueueDestroy failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_command_list_create(void* context, void* device, void** command_list) {
	if (!command_list) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "command list pointer is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

#if defined(L0_STUB)
	*command_list = nullptr;
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!context || !device) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "context/device handle is null");
		*command_list = nullptr;
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_command_list_desc_t desc = {};
	desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
	desc.pNext = nullptr;
	desc.commandQueueGroupOrdinal = 0;

	ze_command_list_handle_t handle = nullptr;
	ze_result_t result = zeCommandListCreate(
		reinterpret_cast<ze_context_handle_t>(context),
		reinterpret_cast<ze_device_handle_t>(device),
		&desc,
		&handle);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeCommandListCreate failed");
		*command_list = nullptr;
		return static_cast<int32_t>(result);
	}

	*command_list = handle;
	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_command_list_reset(void* command_list) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!command_list) {
		return static_cast<int32_t>(ZE_RESULT_SUCCESS);
	}

	ze_result_t result = zeCommandListReset(reinterpret_cast<ze_command_list_handle_t>(command_list));
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeCommandListReset failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_command_list_close(void* command_list) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!command_list) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "command list handle is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_result_t result = zeCommandListClose(reinterpret_cast<ze_command_list_handle_t>(command_list));
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeCommandListClose failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_command_list_destroy(void* command_list) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!command_list) {
		return static_cast<int32_t>(ZE_RESULT_SUCCESS);
	}

	ze_result_t result = zeCommandListDestroy(reinterpret_cast<ze_command_list_handle_t>(command_list));
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeCommandListDestroy failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_command_queue_execute(void* queue, void* command_list) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!queue || !command_list) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "queue or command list is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_command_list_handle_t list = reinterpret_cast<ze_command_list_handle_t>(command_list);
	ze_result_t result = zeCommandQueueExecuteCommandLists(
		reinterpret_cast<ze_command_queue_handle_t>(queue),
		1,
		&list,
		nullptr);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeCommandQueueExecuteCommandLists failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_command_queue_synchronize(void* queue, uint64_t timeout_ns) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!queue) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "queue is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_result_t result = zeCommandQueueSynchronize(
		reinterpret_cast<ze_command_queue_handle_t>(queue),
		timeout_ns);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeCommandQueueSynchronize failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_module_create(void* context, void* device, const uint8_t* module_bytes, uint32_t size, uint32_t module_format, void** module, char* build_log, uint32_t build_log_size) {
	if (!module) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "module pointer is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

#if defined(L0_STUB)
	*module = nullptr;
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (build_log && build_log_size > 0) {
		build_log[0] = '\0';
	}

	if (!context || !device || !module_bytes || size == 0) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "context/device/module bytes is null");
		*module = nullptr;
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_module_format_t resolved_format = ZE_MODULE_FORMAT_IL_SPIRV;
	if (!try_get_module_format(module_format, resolved_format)) {
		*module = nullptr;
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_ARGUMENT);
	}

	ze_module_desc_t desc = {};
	desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
	desc.pNext = nullptr;
	desc.format = resolved_format;
	desc.inputSize = size;
	desc.pInputModule = module_bytes;
	desc.pBuildFlags = "";

	ze_module_handle_t handle = nullptr;
	ze_module_build_log_handle_t log = nullptr;
	ze_result_t result = zeModuleCreate(
		reinterpret_cast<ze_context_handle_t>(context),
		reinterpret_cast<ze_device_handle_t>(device),
		&desc,
		&handle,
		&log);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeModuleCreate failed");
	}

	if (log && build_log && build_log_size > 0) {
		size_t logSize = build_log_size;
		zeModuleBuildLogGetString(log, &logSize, build_log);
		build_log[build_log_size - 1] = '\0';
		zeModuleBuildLogDestroy(log);
	}

	if (result != ZE_RESULT_SUCCESS) {
		*module = nullptr;
		return static_cast<int32_t>(result);
	}

	*module = handle;
	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_module_destroy(void* module) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!module) {
		return static_cast<int32_t>(ZE_RESULT_SUCCESS);
	}

	ze_result_t result = zeModuleDestroy(reinterpret_cast<ze_module_handle_t>(module));
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeModuleDestroy failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_kernel_create(void* module, const char* name, void** kernel) {
	if (!kernel) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "kernel pointer is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

#if defined(L0_STUB)
	*kernel = nullptr;
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!module || !name) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "module/name is null");
		*kernel = nullptr;
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_kernel_desc_t desc = {};
	desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
	desc.pNext = nullptr;
	desc.flags = 0;
	desc.pKernelName = name;

	ze_kernel_handle_t handle = nullptr;
	ze_result_t result = zeKernelCreate(reinterpret_cast<ze_module_handle_t>(module), &desc, &handle);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeKernelCreate failed");
		*kernel = nullptr;
		return static_cast<int32_t>(result);
	}

	*kernel = handle;
	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_kernel_destroy(void* kernel) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!kernel) {
		return static_cast<int32_t>(ZE_RESULT_SUCCESS);
	}

	ze_result_t result = zeKernelDestroy(reinterpret_cast<ze_kernel_handle_t>(kernel));
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeKernelDestroy failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_kernel_set_group_size(void* kernel, uint32_t group_x, uint32_t group_y, uint32_t group_z) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!kernel) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "kernel is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_result_t result = zeKernelSetGroupSize(
		reinterpret_cast<ze_kernel_handle_t>(kernel),
		group_x,
		group_y,
		group_z);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeKernelSetGroupSize failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_kernel_set_arg_value(void* kernel, uint32_t index, size_t size, const void* value) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!kernel) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "kernel is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_result_t result = zeKernelSetArgumentValue(
		reinterpret_cast<ze_kernel_handle_t>(kernel),
		index,
		size,
		value);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeKernelSetArgumentValue failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_kernel_set_arg_mem(void* kernel, uint32_t index, const void* ptr) {
	return lz_kernel_set_arg_value(kernel, index, sizeof(void*), &ptr);
}

int32_t lz_command_list_append_launch_kernel(void* command_list, void* kernel, uint32_t group_x, uint32_t group_y, uint32_t group_z) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!command_list || !kernel) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "command list/kernel is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_group_count_t groupCount = { group_x, group_y, group_z };
	ze_result_t result = zeCommandListAppendLaunchKernel(
		reinterpret_cast<ze_command_list_handle_t>(command_list),
		reinterpret_cast<ze_kernel_handle_t>(kernel),
		&groupCount,
		nullptr,
		0,
		nullptr);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeCommandListAppendLaunchKernel failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_command_list_append_barrier(void* command_list) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!command_list) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "command list is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_result_t result = zeCommandListAppendBarrier(reinterpret_cast<ze_command_list_handle_t>(command_list), nullptr, 0, nullptr);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeCommandListAppendBarrier failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_command_list_append_memory_copy(void* command_list, void* dst, const void* src, size_t size) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!command_list || !dst || !src) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "command list/dst/src is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_result_t result = zeCommandListAppendMemoryCopy(
		reinterpret_cast<ze_command_list_handle_t>(command_list),
		dst,
		src,
		size,
		nullptr,
		0,
		nullptr);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeCommandListAppendMemoryCopy failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_usm_alloc_shared(void* context, void* device, size_t size, size_t alignment, void** ptr) {
	if (!ptr) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "ptr is null");
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

#if defined(L0_STUB)
	*ptr = nullptr;
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!context || !device) {
		set_last_error(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "context/device is null");
		*ptr = nullptr;
		return static_cast<int32_t>(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
	}

	ze_device_mem_alloc_desc_t deviceDesc = {};
	deviceDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
	deviceDesc.pNext = nullptr;
	deviceDesc.ordinal = 0;
	deviceDesc.flags = 0;

	ze_host_mem_alloc_desc_t hostDesc = {};
	hostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
	hostDesc.pNext = nullptr;
	hostDesc.flags = 0;

	void* outPtr = nullptr;
	ze_result_t result = zeMemAllocShared(
		reinterpret_cast<ze_context_handle_t>(context),
		&deviceDesc,
		&hostDesc,
		size,
		alignment,
		reinterpret_cast<ze_device_handle_t>(device),
		&outPtr);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeMemAllocShared failed");
		*ptr = nullptr;
		return static_cast<int32_t>(result);
	}

	*ptr = outPtr;
	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

int32_t lz_usm_free(void* context, void* ptr) {
#if defined(L0_STUB)
	set_last_error(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE, "Level Zero loader not available");
	return static_cast<int32_t>(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
#else
	if (!context || !ptr) {
		return static_cast<int32_t>(ZE_RESULT_SUCCESS);
	}

	ze_result_t result = zeMemFree(reinterpret_cast<ze_context_handle_t>(context), ptr);
	if (result != ZE_RESULT_SUCCESS) {
		set_last_error(result, "zeMemFree failed");
		return static_cast<int32_t>(result);
	}

	g_last_result = static_cast<int32_t>(result);
	g_last_error.clear();
	return static_cast<int32_t>(result);
#endif
}

const char* lz_get_last_error() {
	return g_last_error.c_str();
}

int32_t lz_get_last_result() {
	return g_last_result;
}

void lz_clear_error() {
	g_last_error.clear();
	g_last_result = 0;
}

}
