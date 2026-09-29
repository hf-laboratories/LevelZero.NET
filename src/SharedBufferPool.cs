using System.Collections.Concurrent;

namespace LevelZero;

/// <summary>
/// Thread-safe object pool for Level Zero <see cref="SharedBuffer{T}"/> allocations.
/// Implements PAT-CS-242 (Persistent Acceleration Buffer Pool) to eliminate ephemeral
/// USM allocation and deallocation churn inside iterative GPU execution loops.
/// </summary>
public sealed class SharedBufferPool<T> : IDisposable where T : unmanaged
{
    private readonly ComputeDevice _device;
    private readonly ConcurrentBag<SharedBuffer<T>> _buffers = new();
    private bool _disposed;

    public SharedBufferPool(ComputeDevice device)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));
    }

    /// <summary>
    /// Rents a buffer with at least <paramref name="minCapacity"/> elements.
    /// Reuses a pooled buffer if available; otherwise allocates a new shared buffer on the device.
    /// </summary>
    public SharedBuffer<T> Rent(int minCapacity)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        while (_buffers.TryTake(out SharedBuffer<T>? candidate))
        {
            if (candidate.Count >= minCapacity)
            {
                return candidate;
            }

            // Candidate is too small for requested capacity; discard it
            candidate.Dispose();
        }

        return _device.AllocShared<T>(minCapacity);
    }

    /// <summary>
    /// Returns a previously rented buffer to the pool for reuse.
    /// </summary>
    public void Return(SharedBuffer<T> buffer)
    {
        ArgumentNullException.ThrowIfNull(buffer);

        if (_disposed)
        {
            buffer.Dispose();
            return;
        }

        _buffers.Add(buffer);
    }

    /// <summary>
    /// Rents a buffer wrapped in a scoped struct that automatically returns it to the pool on disposal.
    /// </summary>
    public RentedBufferScope RentScoped(int minCapacity)
    {
        SharedBuffer<T> buffer = Rent(minCapacity);
        return new RentedBufferScope(this, buffer);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        while (_buffers.TryTake(out SharedBuffer<T>? buf))
        {
            buf.Dispose();
        }
    }

    /// <summary>
    /// Disposable scope for pooled buffer lifetime management with C# <c>using</c> declarations.
    /// </summary>
    public readonly struct RentedBufferScope : IDisposable
    {
        private readonly SharedBufferPool<T> _pool;
        public SharedBuffer<T> Buffer { get; }

        public RentedBufferScope(SharedBufferPool<T> pool, SharedBuffer<T> buffer)
        {
            _pool = pool;
            Buffer = buffer;
        }

        public void Dispose()
        {
            _pool.Return(Buffer);
        }
    }
}
