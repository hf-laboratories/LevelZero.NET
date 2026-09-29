using System;

namespace LevelZero;

/// <summary>
/// Describes a multi-dimensional tensor's shape, strides, and offset.
/// Useful for managing slices, sub-grids, and views (e.g. for key/value caches) during GPU execution.
/// </summary>
public sealed class TensorDescriptor
{
    /// <summary>The size of each dimension in the tensor.</summary>
    public int[] Shape { get; }

    /// <summary>The stride (element step size) for each dimension.</summary>
    public int[] Strides { get; }

    /// <summary>The base offset index in elements from the start of the underlying allocation.</summary>
    public int Offset { get; }

    /// <summary>Total number of elements represented by this descriptor.</summary>
    public int Size { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorDescriptor"/> class.
    /// </summary>
    public TensorDescriptor(int[] shape, int[]? strides = null, int offset = 0)
    {
        Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        Offset = offset;

        if (strides != null)
        {
            if (strides.Length != shape.Length)
            {
                throw new ArgumentException("Strides length must match shape length.", nameof(strides));
            }
            Strides = strides;
        }
        else
        {
            Strides = CalculateDefaultStrides(shape);
        }

        Size = CalculateTotalSize(shape);
    }

    private static int[] CalculateDefaultStrides(int[] shape)
    {
        int[] strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int CalculateTotalSize(int[] shape)
    {
        int size = 1;
        foreach (int dim in shape)
        {
            size *= dim;
        }
        return size;
    }
}
