using System.Runtime.CompilerServices;
using LevelZero.Native;

namespace LevelZero;

/// <summary>
/// Typed wrapper around a Level Zero USM shared-memory allocation.
/// Provides safe Write/Read operations for copying data to/from the GPU.
/// </summary>
public sealed class SharedBuffer<T> : IDisposable where T : unmanaged
{
    private readonly IntPtr _contextHandle;
    private IntPtr _ptr;
    private readonly int _count;
    private bool _disposed;

    internal SharedBuffer(IntPtr contextHandle, IntPtr ptr, int count)
    {
        _contextHandle = contextHandle;
        _ptr = ptr;
        _count = count;
    }

    /// <summary>Raw pointer to the USM allocation. Valid for kernel argument binding.</summary>
    public IntPtr Pointer => !_disposed ? _ptr : throw new ObjectDisposedException(nameof(SharedBuffer<T>));

    /// <summary>Number of elements of type T in this buffer.</summary>
    public int Count => _count;

    /// <summary>Total byte length of this buffer.</summary>
    public int ByteLength => _count * Unsafe.SizeOf<T>();

    /// <summary>Copies data from a managed span into the shared buffer.</summary>
    public void Write(ReadOnlySpan<T> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (source.Length > _count)
            throw new ArgumentException($"Source length {source.Length} exceeds buffer capacity {_count}");

        unsafe
        {
            var dest = new Span<T>((void*)_ptr, _count);
            source.CopyTo(dest);
        }
    }

    /// <summary>Copies data from a managed array into the shared buffer.</summary>
    public void Write(T[] source) => Write(source.AsSpan());

    /// <summary>Copies the full buffer contents into a managed span.</summary>
    public void ReadTo(Span<T> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < _count)
            throw new ArgumentException($"Destination length {destination.Length} is smaller than buffer count {_count}");

        unsafe
        {
            var src = new ReadOnlySpan<T>((void*)_ptr, _count);
            src.CopyTo(destination);
        }
    }

    /// <summary>Reads the full buffer into a new managed array.</summary>
    public T[] ToArray()
    {
        var result = new T[_count];
        ReadTo(result);
        return result;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ptr != IntPtr.Zero)
        {
            LevelZeroNative.lz_usm_free(_contextHandle, _ptr);
            _ptr = IntPtr.Zero;
        }
    }
}
