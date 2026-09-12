#pragma once

/* <editor-fold desc="MIT License">

Copyright(c) 2018 Robert Osfield

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <deque>
#include <map>
#include <memory>
#include <vector>

#include <vsg/core/Object.h>
#include <vsg/state/BufferInfo.h>
#include <vsg/vk/ResourceRequirements.h>

namespace vsg
{

    /// MemoryBufferPools manages a pool of vsg::DeviceMemory and vsg::Buffer that use them.
    /// Methods are provided for getting Buffer from the pool, sharing memory to make better use of device memory.
    class VSG_DECLSPEC MemoryBufferPools : public Inherit<Object, MemoryBufferPools>
    {
    public:
        MemoryBufferPools(const std::string& name, ref_ptr<Device> in_device, const ResourceRequirements& in_resourceRequirements = {});

        std::string name;
        ref_ptr<Device> device;
        VkDeviceSize minimumBufferSize = 16 * 1024 * 1024;
        VkDeviceSize minimumDeviceMemorySize = 16 * 1024 * 1024;

        /// Ratio of available device memory that can be allocated.
        /// Ratios less than 1.0 require VK_EXT_memory_budget extension to be supported.
        /// Ratio of 1.0 (or greater) will switch off checks for available memory and keep allocating till Vulkan memory allocations fail.
        double allocatedMemoryLimit = 1.0;

        /// throw vsg::Exception when reserveMemory() fails to allocated memory on device.
        bool throwOutOfDeviceMemoryException = true;

        /// hint whether the compile traversal should call MemoryBufferPools::reserve(requirements);
        bool compileTraversalUseReserve = true;

        VkDeviceSize computeMemoryTotalAvailable() const;
        VkDeviceSize computeMemoryTotalReserved() const;
        VkDeviceSize computeBufferTotalAvailable() const;
        VkDeviceSize computeBufferTotalReserved() const;

        ref_ptr<BufferInfo> reserveBuffer(VkDeviceSize totalSize, VkDeviceSize alignment, VkBufferUsageFlags bufferUsageFlags, VkSharingMode sharingMode, VkMemoryPropertyFlags memoryProperties);

        using DeviceMemoryOffset = std::pair<ref_ptr<DeviceMemory>, VkDeviceSize>;
        DeviceMemoryOffset reserveMemory(VkMemoryRequirements memRequirements, VkMemoryPropertyFlags memoryProperties, void* pNextAllocInfo = nullptr);

        VkResult reserve(ResourceRequirements& requirements);

        void report(LogOutput& out) const;

        /// What this pool has committed, what is free inside it, and how
        /// fragmented that free space is.
        ///
        /// Added for VulkanGIS VGIS-509. The pool never returns an emptied
        /// block to the driver, so "committed" tracks everywhere the camera has
        /// BEEN rather than what is resident -- reporting a single number made
        /// half of 7.8 GB look real when it was empty space inside held blocks.
        /// And `largestContiguous` is the number that explains an allocation
        /// being refused while plenty is nominally free; nothing exposed it.
        ///
        /// Aggregated here rather than by exposing the block vectors, so the
        /// walk happens under the pool's own mutex.
        struct PoolStats
        {
            std::size_t blocks = 0;          ///< committed blocks
            VkDeviceSize totalSize = 0;      ///< bytes committed to the driver
            VkDeviceSize totalAvailable = 0; ///< free bytes inside those blocks
            VkDeviceSize largestContiguous = 0; ///< biggest single run that is free

            /// How many blocks hold NOTHING at all.
            ///
            /// The number that decides whether returning empty blocks to the
            /// driver is worth anything. 544 MB live across 261 blocks is
            /// either ~24 blocks packed full and ~237 returnable, or 2 MB
            /// pinned in every one of them and NOTHING returnable. The
            /// aggregate cannot tell those apart; this can.
            std::size_t emptyBlocks = 0;

            /// Blocks holding less than an eighth of their capacity. A block
            /// that is nearly empty is one relocation away from returnable,
            /// which is what decides whether defragmentation would pay.
            std::size_t nearlyEmptyBlocks = 0;

            /// Bytes held in blocks that are neither empty nor nearly empty.
            VkDeviceSize bytesInBusyBlocks = 0;
        };

        /// One reuse-compatible class of blocks.
        ///
        /// WHY PER CLASS. trimEmptyBlocks() applies its cushion per class, not
        /// per pool, because a block can only satisfy a request it matches. The
        /// floor that creates is therefore `keepFreeBlocks x classes`, and a
        /// total cannot show that: it reports "72 empty blocks" whether that is
        /// one class badly behaved or eighteen classes each holding their four.
        ///
        /// Measured in VulkanGIS the moment tiles stopped appearing: 1.56 GB
        /// standing in 72 empty blocks, 27% of everything committed, on a GPU
        /// already at 85.1% of its cap. 72 is exactly 18 x 4. Whether that was
        /// the cushion or merely blocks not yet aged out is the question this
        /// exists to answer, and nothing reported it.
        struct ClassStats
        {
            /// memoryTypeBits for the device pool, usage flags for the buffer
            /// pool. Reported raw so a reader can tell classes apart without
            /// this header having to name every Vulkan flag.
            uint64_t keyA = 0;
            /// alignment for the device pool, sharing mode for the buffer pool.
            uint64_t keyB = 0;

            std::size_t blocks = 0;
            std::size_t emptyBlocks = 0;
            VkDeviceSize totalSize = 0;
            /// Bytes sitting in the empty blocks of this class.
            VkDeviceSize emptyBytes = 0;
            /// Bytes actually reserved by live allocations in this class.
            ///
            /// totalSize - reservedBytes - emptyBytes is the number VGIS-545
            /// needs and nothing reported: free space STRANDED INSIDE BUSY
            /// BLOCKS, which the empty-block trim cannot reach by definition
            /// and which is what grows when a pool churns payloads of
            /// changing sizes. Measured 2026-09-12: committed memory rose
            /// 32-48 MB per heavy pass with the scene constant and every
            /// block non-empty, so the trim had nothing to return and the
            /// aggregate could not say where the bytes went.
            VkDeviceSize reservedBytes = 0;
        };

        /// Per-class breakdown of the DeviceMemory pool, keyed by the pair
        /// reserveMemory() matches on: (memoryTypeBits, alignment).
        std::vector<ClassStats> deviceMemoryClasses() const;

        /// Per-class breakdown of the Buffer pool, keyed by the pair
        /// reserveBuffer() matches on: (usage, sharingMode).
        std::vector<ClassStats> bufferClasses() const;

        /// Stats for the DeviceMemory blocks (the device-local pool).
        PoolStats deviceMemoryStats() const;

        /// Stats for the Buffer blocks.
        PoolStats bufferStats() const;

        /// What a trim gave back.
        struct TrimResult
        {
            std::size_t blocksReleased = 0;
            VkDeviceSize bytesReleased = 0;
            /// True when nonBlocking was asked for and somebody else held the
            /// lock, so nothing was examined. Not an error: a block that is
            /// empty this frame is still empty on the next call.
            bool skipped = false;
        };

        /// Hand blocks that hold nothing back to the driver.
        ///
        /// WHY THIS EXISTS. Until now `push_back` was the only operation ever
        /// performed on either pool vector: a block, once created, was kept for
        /// the life of the device. Measured in VulkanGIS after ten zoom legs and
        /// eight pans over one city: 269 blocks, 6037 MB committed, 681 MB
        /// actually in use, and 207 of those blocks holding NOTHING AT ALL. The
        /// 4.6 GB in them still counted against the GPU budget, so the driver
        /// refused 392 subsequent allocations and the scene stopped loading
        /// content entirely. Restarting the process fixed it, which is the
        /// signature of a pool that only grows.
        ///
        /// WHY THE FRAME DELAY, and do not remove it. Dropping the pool's
        /// reference is NOT sufficient on its own. Reference counting only
        /// proves no C++ object still points at the block; a command buffer
        /// already submitted can still be reading a VkBuffer bound to that
        /// VkDeviceMemory after the last BufferInfo naming it has gone. So a
        /// block is only released once it has been continuously empty for
        /// `minAgeFrames` frames, which must exceed the number of frames the
        /// application keeps in flight.
        ///
        /// Safe to call once per frame. Must be called on a thread that is not
        /// concurrently recording, and `frame` must increase monotonically.
        ///
        /// \param frame          the current frame number
        /// \param minAgeFrames   how long a block must have been empty
        /// \return what was released
        ///
        /// WHY THE CUSHION. Releasing a block the instant it empties turns the
        /// pool into a thrash loop: the block goes back to the driver and
        /// vkAllocateMemory is called for the same 16 MB a few frames later.
        /// keepFreeBlocks retains that many empty blocks as spare capacity, so
        /// ordinary churn is absorbed inside the pool and only a sustained
        /// surplus is handed back.
        ///
        /// The cushion is PER REUSE-COMPATIBLE CLASS, not per pool. A block can
        /// only satisfy a request it matches -- reserveMemory on memoryTypeBits
        /// and alignment, reserveBuffer on usage and sharing mode -- so a global
        /// count would keep four blocks that cannot serve the request being made
        /// while freeing and reallocating the one class actually in use.
        ///
        /// WHY nonBlocking EXISTS, and why the default is the safe one.
        /// This function takes _mutex, and _mutex is the ALLOCATION lock:
        /// reserveMemory holds it across DeviceMemory::create, which is
        /// vkAllocateMemory, and reserveBuffer holds it across
        /// Buffer::compile. Both run on compile threads and both can sit in
        /// the driver for tens of milliseconds when video memory is under
        /// pressure. A caller on the RENDER thread that blocks here therefore
        /// pays a compile thread's driver time out of its own frame.
        ///
        /// rocky calls this once per frame from VSGContextImpl::update(), so
        /// it passes true. Skipping is free: a block that is empty now is
        /// still empty next frame, the age clock is the viewer's rendered
        /// frame count rather than a count of calls, and the cushion means
        /// nothing is urgent. This mirrors the try_lock already used for the
        /// garbage collector a few lines earlier in that same function, which
        /// took frames over 50 ms in a heavy zoom run from 70 to none.
        ///
        /// \param frame          the current frame number
        /// \param minAgeFrames   how long a block must have been empty
        /// \param keepFreeBlocks empty blocks to retain per compatibility class
        /// \param nonBlocking    return immediately if an allocation holds the lock
        /// \return what was released, or skipped = true
        TrimResult trimEmptyBlocks(uint64_t frame, uint64_t minAgeFrames = 3, std::size_t keepFreeBlocks = 0,
                                   bool nonBlocking = false);

    protected:
        mutable std::mutex _mutex;

        // transfer data settings
        using MemoryPools = std::vector<ref_ptr<DeviceMemory>>;
        MemoryPools memoryPools;

        using BufferPools = std::vector<ref_ptr<Buffer>>;
        BufferPools bufferPools;

        /// Frame on which a block was FIRST seen empty, keyed by block.
        /// Cleared for any block that is used again, so the age only counts
        /// uninterrupted emptiness. See trimEmptyBlocks().
        std::map<const Object*, uint64_t> _emptySince;
    };
    VSG_type_name(vsg::MemoryBufferPools);

} // namespace vsg
