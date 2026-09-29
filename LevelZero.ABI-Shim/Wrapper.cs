using System;
using System.Runtime.InteropServices;
using System.Text;

namespace LevelZero.Interop
{
    /// <summary>
    /// P/Invoke wrapper for LevelZero C++ library
    /// Generated automatically by HFLabs C++ ABI Wrapper Generator v2.0
    /// 
    /// This class provides managed access to native LevelZero functionality
    /// through a C ABI shim layer for maximum compatibility.
    /// </summary>
    public static class LevelZeroNative
    {
        // Library name - adjust for your platform
#if UNITY_IPHONE && !UNITY_EDITOR
        private const string LIB = "__Internal";
#elif UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        private const string LIB = "LevelZero.dll";
#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
        private const string LIB = "libLevelZero.dylib";
#elif UNITY_STANDALONE_LINUX || UNITY_EDITOR_LINUX
        private const string LIB = "libLevelZero.so";
#elif NETFRAMEWORK || NETCOREAPP || NET5_0_OR_GREATER
#if WINDOWS
        private const string LIB = "LevelZero.dll";
#elif OSX
        private const string LIB = "libLevelZero.dylib";
#else
        private const string LIB = "libLevelZero.so";
#endif
#else
        private const string LIB = "LevelZero";
#endif

        // ========== MEMORY MANAGEMENT ==========
        
        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern void shim_free_string(IntPtr str);
        
        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern void shim_free_wstring(IntPtr str);
        
        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern void shim_free_u8string(IntPtr str);
        
        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern void shim_free_u16string(IntPtr str);
        
        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        private static extern void shim_free_u32string(IntPtr str);

        /// <summary>
        /// Converts an IntPtr to a string and frees the native memory
        /// </summary>
        public static string PtrToStringAndFree(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero) return null;
            try
            {
                string result = Marshal.PtrToStringAnsi(ptr);
                shim_free_string(ptr);
                return result;
            }
            catch
            {
                shim_free_string(ptr);
                return null;
            }
        }

        /// <summary>
        /// Converts an IntPtr to a wide string and frees the native memory
        /// </summary>
        public static string PtrToWStringAndFree(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero) return null;
            try
            {
                string result = Marshal.PtrToStringUni(ptr);
                shim_free_wstring(ptr);
                return result;
            }
            catch
            {
                shim_free_wstring(ptr);
                return null;
            }
        }

        /// <summary>
        /// Converts an IntPtr to a UTF-8 string and frees the native memory
        /// </summary>
        public static string PtrToUtf8StringAndFree(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero) return null;
            try
            {
                string result = Marshal.PtrToStringUTF8(ptr);
                shim_free_u8string(ptr);
                return result;
            }
            catch
            {
                shim_free_u8string(ptr);
                return null;
            }
        }

        // ========== NOTES AND BEST PRACTICES ==========
        /*
         * MEMORY MANAGEMENT:
         * - All string-returning functions have *_str variants that automatically handle memory cleanup
         * - For raw IntPtr returns, use the PtrToStringAndFree helpers or manually call shim_free_* functions
         * - Complex types (containers, smart pointers) are opaque handles - implement wrapper classes as needed
         * 
         * COMPLEX TYPE HANDLING:
         * - std::vector, std::map, etc. -> IntPtr (implement custom wrapper methods)
         * - std::unique_ptr, std::shared_ptr -> IntPtr (implement reference counting if needed)
         * - std::thread, std::mutex -> IntPtr (implement thread-safe wrappers)
         * - std::optional, std::variant -> IntPtr (implement value extraction methods)
         * - Ranges, Views, Coroutines -> IntPtr (implement modern C++ feature wrappers)
         * 
         * STRING ENCODING:
         * - std::string -> UTF-8/ANSI depending on system
         * - std::wstring -> UTF-16 (Windows) / UTF-32 (Unix)
         * - std::u8string -> UTF-8 explicit
         * - std::u16string -> UTF-16 explicit  
         * - std::u32string -> UTF-32 explicit
         * 
         * THREADING SAFETY:
         * - Generated wrappers are not inherently thread-safe
         * - Use appropriate synchronization for std::atomic, std::mutex wrapper objects
         * - Consider using concurrent collections in C# for better performance
         */
    }

    /// <summary>
    /// Example usage pattern for complex types:
    /// 
    /// // For containers like std::vector:
    /// IntPtr vectorHandle = SomeFunction_that_returns_vector();
    /// // Implement helper methods to interact with the vector
    /// // int GetVectorSize(IntPtr handle);
    /// // T GetVectorElement(IntPtr handle, int index);
    /// // void AddToVector(IntPtr handle, T element);
    /// 
    /// // For smart pointers:
    /// IntPtr smartPtrHandle = SomeFunction_that_returns_unique_ptr();
    /// // Implement wrapper methods:
    /// // IntPtr GetRawPointer(IntPtr smartPtr);
    /// // void ResetSmartPtr(IntPtr smartPtr);
    /// // bool IsValidSmartPtr(IntPtr smartPtr);
    /// </summary>
    public static class LevelZeroExamples
    {
        // Add your custom wrapper methods for complex types here
    }
}

