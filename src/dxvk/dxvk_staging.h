#pragma once

#include <queue>

#include "dxvk_buffer.h"

namespace dxvk {
  
  class DxvkDevice;

  /**
   * \brief Staging data allocator
   *
   * Allocates buffer slices for resource uploads,
   * while trying to keep the number of allocations
   * but also the amount of allocated memory low.
   */
  class DxvkStagingDataAlloc {
    constexpr static VkDeviceSize MaxBufferSize  = 1 << 25; // 32 MiB
    constexpr static uint32_t     MaxBufferCount = 2;
  public:

    DxvkStagingDataAlloc(const Rc<DxvkDevice>& device);

    ~DxvkStagingDataAlloc();

    /**
     * \brief Alloctaes a staging buffer slice
     * 
     * \param [in] align Alignment of the allocation
     * \param [in] size Size of the allocation
     * \returns Staging buffer slice
     */
    DxvkBufferSlice alloc(VkDeviceSize align, VkDeviceSize size);

    /**
     * \brief Deletes all staging buffers
     * 
     * Destroys allocated buffers and
     * releases all buffer memory.
     */
    void trim();

  private:

    Rc<DxvkDevice>  m_device;
    Rc<DxvkBuffer>  m_buffer;
    VkDeviceSize    m_offset = 0;

    std::queue<Rc<DxvkBuffer>> m_buffers;

    Rc<DxvkBuffer> createBuffer(VkDeviceSize size);

  };

  /**
   * \brief Staging buffer
   *
   * Provides a simple linear staging buffer
   * allocator for data uploads.
   */
  class DxvkStagingBuffer {

  public:

    /**
     * \brief Creates staging buffer
     *
     * \param [in] device DXVK device
     * \param [in] size Buffer size
     */
    DxvkStagingBuffer(
      const Rc<DxvkDevice>&     device,
            VkDeviceSize        size);

    /**
     * \brief Frees staging buffer
     */
    ~DxvkStagingBuffer();

    /**
     * \brief Allocates staging buffer memory
     *
     * Tries to suballocate from existing buffer,
     * or creates a new buffer if necessary.
     * \param [in] align Minimum alignment
     * \param [in] size Number of bytes to allocate
     * \returns Allocated slice
     */
    DxvkBufferSlice alloc(VkDeviceSize align, VkDeviceSize size);

    /**
     * \brief Resets staging buffer and allocator
     */
    void reset();

  private:

    Rc<DxvkDevice>  m_device;
    Rc<DxvkBuffer>  m_buffer;
    VkDeviceSize    m_offset;
    VkDeviceSize    m_size;

  };

}
