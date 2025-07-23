use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter};

pub struct VulkanBuffer<T: ?Sized> {
    pub(crate) device_buffer: Subbuffer<T>,
}

impl<T> VulkanBuffer<[T]>
where
    T: BufferContents,
{
    pub fn new<I>(
        memory_allocator: Arc<dyn MemoryAllocator>,
        data: I,
        buffer_usage: BufferUsage,
    ) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let device_buffer_info = BufferCreateInfo {
            usage: buffer_usage,
            ..Default::default()
        };

        let device_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            device_buffer_info,
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.into_iter(),
        )
        .unwrap();

        Self { device_buffer }
    }
}

impl<T> VulkanBuffer<T>
where
    T: BufferContents,
{
    pub fn new_with_data(
        memory_allocator: Arc<dyn MemoryAllocator>,
        data: T,
        buffer_usage: BufferUsage,
    ) -> Self {
        let device_buffer_info = BufferCreateInfo {
            usage: buffer_usage,
            ..Default::default()
        };

        let device_buffer = Buffer::from_data(
            memory_allocator.clone(),
            device_buffer_info,
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data,
        )
        .unwrap();

        Self { device_buffer }
    }
}
