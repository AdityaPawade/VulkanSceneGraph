/* <editor-fold desc="MIT License">

Copyright(c) 2018 Robert Osfield

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <vsg/io/Logger.h>
#include <vsg/vk/MemoryBufferPools.h>

#include <algorithm>
#include <chrono>
#include <vector>

using namespace vsg;

MemoryBufferPools::MemoryBufferPools(const std::string& in_name, ref_ptr<Device> in_device, const ResourceRequirements& in_resourceRequirements) :
    name(in_name),
    device(in_device),
    minimumBufferSize(in_resourceRequirements.minimumBufferSize),
    minimumDeviceMemorySize(in_resourceRequirements.minimumDeviceMemorySize)
{
}

VkDeviceSize MemoryBufferPools::computeMemoryTotalAvailable() const
{
    std::scoped_lock<std::mutex> lock(_mutex);

    VkDeviceSize totalAvailableSize = 0;
    for (auto& deviceMemory : memoryPools)
    {
        totalAvailableSize += deviceMemory->totalAvailableSize();
    }
    return totalAvailableSize;
}

// See the header for why these exist (VulkanGIS VGIS-509). One walk under the
// pool's own mutex, so a caller cannot see a half-updated set of blocks.
//
// largestContiguous is the MAXIMUM over blocks, not a sum: an allocation has to
// fit inside ONE block, so the biggest single free run is what decides whether
// the next reserve() succeeds. Summing the per-block figures would produce a
// number that looks healthy at exactly the moment allocations start failing.
MemoryBufferPools::PoolStats MemoryBufferPools::deviceMemoryStats() const
{
    std::scoped_lock<std::mutex> lock(_mutex);

    PoolStats s;
    s.blocks = memoryPools.size();
    for (auto& deviceMemory : memoryPools)
    {
        const VkDeviceSize used = deviceMemory->totalReservedSize();
        const VkDeviceSize avail = deviceMemory->totalAvailableSize();
        s.totalSize += used + avail;
        s.totalAvailable += avail;
        s.largestContiguous = std::max(s.largestContiguous,
            static_cast<VkDeviceSize>(deviceMemory->maximumAvailableSpace()));

        if (used == 0) ++s.emptyBlocks;
        else if (used * 8 < used + avail) ++s.nearlyEmptyBlocks;
        else s.bytesInBusyBlocks += used;
    }
    return s;
}

// See the header for the measurement that motivated this and for why the frame
// delay is not optional.
//
// The two pools are walked with one shared helper because the rule is identical
// for both: a block that reports nothing reserved, and has reported that for
// long enough that no submitted work can still be reading it, is dropped.
//
// Dropping means erasing the pool's ref_ptr. That does not necessarily destroy
// anything: if a BufferInfo somewhere still names the block it stays alive and
// simply stops being handed out. It is destroyed when the last reference goes,
// which is what makes this safe to do from the pool without knowing who else
// holds one.
MemoryBufferPools::TrimResult MemoryBufferPools::trimEmptyBlocks(uint64_t frame, uint64_t minAgeFrames, std::size_t keepFreeBlocks)
{
    std::scoped_lock<std::mutex> lock(_mutex);

    TrimResult result;

    // Age a block, and say whether it has now been empty long enough to go.
    // Every block is aged on every call, including ones the cushion will
    // protect -- otherwise a pool sitting just above the cushion would have
    // the clock reset on it forever and would never trim at all.
    const auto age = [&](const Object* block, bool isEmpty) {
        if (!isEmpty)
        {
            _emptySince.erase(block);
            return false;
        }
        auto [itr, inserted] = _emptySince.try_emplace(block, frame);
        // A frame number that went backwards (a restart, a reset counter)
        // would underflow the unsigned subtraction and make every block look
        // ancient, releasing the lot at once. Treat it as freshly empty.
        if (frame < itr->second)
        {
            itr->second = frame;
            return false;
        }
        return !inserted && (frame - itr->second) >= minAgeFrames;
    };

    // Both pools follow the same rule, so they share one walk. keyOf says which
    // blocks can substitute for one another and blockBytes how big one is;
    // those are the only things that differ between the two.
    //
    // WHY THE CUSHION IS PER CLASS AND NOT GLOBAL. A block can only satisfy a
    // request it is compatible with: reserveMemory matches on memoryTypeBits
    // AND alignment, reserveBuffer on usage AND sharing mode. A single global
    // cushion of four is therefore worthless to a workload that cycles through
    // five classes -- once all five are empty the count is five, one over the
    // cushion, so one is freed every cycle and reallocated the moment that
    // class is touched again. The pool would round-trip through
    // vkAllocateMemory forever while holding four blocks that cannot serve the
    // request being made. Counting per class means each class keeps its own
    // spare and a rotating working set is never evicted.
    const auto trimPool = [&](auto& pool, auto&& keyOf, auto&& blockBytes) {
        struct Bucket
        {
            std::size_t empty = 0;
            std::vector<std::pair<uint64_t, std::size_t>> releasable; // (emptySince, index)
        };
        std::map<std::pair<uint64_t, uint64_t>, Bucket> buckets;

        for (std::size_t i = 0; i < pool.size(); ++i)
        {
            const Object* block = pool[i].get();
            const bool isEmpty = pool[i]->totalReservedSize() == 0;

            // Age every block, empty or not, and before the continue below:
            // this is what clears the clock on a block that got used again.
            const bool aged = age(block, isEmpty);
            if (!isEmpty) continue;

            auto& bucket = buckets[keyOf(pool[i])];
            ++bucket.empty;
            if (aged) bucket.releasable.emplace_back(_emptySince[block], i);
        }

        std::vector<bool> drop(pool.size(), false);
        bool anyDropped = false;

        for (auto& entry : buckets)
        {
            Bucket& bucket = entry.second;
            if (bucket.empty <= keepFreeBlocks || bucket.releasable.empty()) continue;

            std::size_t budget = bucket.empty - keepFreeBlocks;
            if (budget > bucket.releasable.size()) budget = bucket.releasable.size();

            // Longest-empty first. Keeping the most recently emptied is
            // deliberate: those are the ones the camera's current position was
            // just cycling, so they are the ones about to be asked for again.
            std::sort(bucket.releasable.begin(), bucket.releasable.end());
            bucket.releasable.resize(budget);

            for (const auto& candidate : bucket.releasable)
            {
                const std::size_t i = candidate.second;
                drop[i] = true;
                anyDropped = true;
                result.bytesReleased += blockBytes(pool[i]);
                ++result.blocksReleased;
                _emptySince.erase(pool[i].get());
            }
        }

        if (!anyDropped) return;

        std::size_t write = 0;
        for (std::size_t i = 0; i < pool.size(); ++i)
        {
            if (!drop[i]) pool[write++] = std::move(pool[i]);
        }
        pool.resize(write);
    };

    // A block reaching here holds nothing, so everything in it is free space;
    // that free total IS the block size.
    trimPool(
        memoryPools,
        [](const ref_ptr<DeviceMemory>& b) {
            const VkMemoryRequirements& r = b->getMemoryRequirements();
            return std::pair<uint64_t, uint64_t>(r.memoryTypeBits, r.alignment);
        },
        [](const ref_ptr<DeviceMemory>& b) { return b->totalAvailableSize(); });

    // USAGE ONLY, deliberately, and it must stay matched to reserveBuffer.
    //
    // reserveBuffer above tests `bufferFromPool->usage == bufferUsageFlags` and
    // never compares sharingMode -- it takes the parameter and ignores it. So
    // two empty buffers differing only in sharing mode are interchangeable to
    // the allocator. Bucketing them apart here would split one reuse class into
    // two and let the cushion pin 2 x keepFreeBlocks where the allocator only
    // ever needed one class's worth, which is precisely the wrong thing to do
    // while the device is refusing allocations.
    //
    // If reserveBuffer ever starts matching on sharingMode, this key must gain
    // it in the same commit. The rule is that the trim keeps spares that can
    // actually serve the next request, and only reserveBuffer defines "can".
    trimPool(
        bufferPools,
        [](const ref_ptr<Buffer>& b) {
            return std::pair<uint64_t, uint64_t>(b->usage, 0);
        },
        [](const ref_ptr<Buffer>& b) { return b->size; });

    return result;
}

// See the header for what these answer. The key pair is exactly what the
// matching reserve() compares on, so two blocks share a class here if and only
// if one could have satisfied a request the other did.
namespace
{
    template<typename POOL, typename KEYOF, typename SIZEOF>
    std::vector<MemoryBufferPools::ClassStats> classify(const POOL& pool, KEYOF keyOf, SIZEOF sizeOf)
    {
        std::map<std::pair<uint64_t, uint64_t>, MemoryBufferPools::ClassStats> byClass;

        for (const auto& block : pool)
        {
            const auto key = keyOf(block);
            auto& c = byClass[key];
            c.keyA = key.first;
            c.keyB = key.second;
            ++c.blocks;

            const VkDeviceSize size = sizeOf(block);
            c.totalSize += size;
            if (block->totalReservedSize() == 0)
            {
                ++c.emptyBlocks;
                c.emptyBytes += size;
            }
        }

        std::vector<MemoryBufferPools::ClassStats> result;
        result.reserve(byClass.size());
        for (const auto& entry : byClass) result.push_back(entry.second);
        return result;
    }
} // namespace

std::vector<MemoryBufferPools::ClassStats> MemoryBufferPools::deviceMemoryClasses() const
{
    std::scoped_lock<std::mutex> lock(_mutex);

    return classify(memoryPools,
        [](const ref_ptr<DeviceMemory>& b) {
            const VkMemoryRequirements& r = b->getMemoryRequirements();
            return std::pair<uint64_t, uint64_t>(r.memoryTypeBits, r.alignment);
        },
        [](const ref_ptr<DeviceMemory>& b) {
            return b->totalReservedSize() + b->totalAvailableSize();
        });
}

std::vector<MemoryBufferPools::ClassStats> MemoryBufferPools::bufferClasses() const
{
    std::scoped_lock<std::mutex> lock(_mutex);

    // Keyed exactly as trimEmptyBlocks keys it, and for the same reason:
    // reserveBuffer matches on usage alone. A report that classified buffers
    // differently from the policy would be describing a pool that does not
    // exist.
    return classify(bufferPools,
        [](const ref_ptr<Buffer>& b) {
            return std::pair<uint64_t, uint64_t>(b->usage, 0);
        },
        [](const ref_ptr<Buffer>& b) { return b->size; });
}

MemoryBufferPools::PoolStats MemoryBufferPools::bufferStats() const
{
    std::scoped_lock<std::mutex> lock(_mutex);

    PoolStats s;
    s.blocks = bufferPools.size();
    for (auto& buffer : bufferPools)
    {
        const VkDeviceSize used = buffer->totalReservedSize();
        s.totalSize += buffer->size;
        s.totalAvailable += buffer->totalAvailableSize();
        s.largestContiguous = std::max(s.largestContiguous,
            static_cast<VkDeviceSize>(buffer->maximumAvailableSpace()));

        if (used == 0) ++s.emptyBlocks;
        else if (used * 8 < buffer->size) ++s.nearlyEmptyBlocks;
        else s.bytesInBusyBlocks += used;
    }
    return s;
}

VkDeviceSize MemoryBufferPools::computeMemoryTotalReserved() const
{
    std::scoped_lock<std::mutex> lock(_mutex);

    VkDeviceSize totalReservedSize = 0;
    for (auto& deviceMemory : memoryPools)
    {
        totalReservedSize += deviceMemory->totalReservedSize();
    }
    return totalReservedSize;
}

VkDeviceSize MemoryBufferPools::computeBufferTotalAvailable() const
{
    std::scoped_lock<std::mutex> lock(_mutex);

    VkDeviceSize totalAvailableSize = 0;
    for (auto& buffer : bufferPools)
    {
        totalAvailableSize += buffer->totalAvailableSize();
    }
    return totalAvailableSize;
}

VkDeviceSize MemoryBufferPools::computeBufferTotalReserved() const
{
    std::scoped_lock<std::mutex> lock(_mutex);

    VkDeviceSize totalReservedSize = 0;
    for (auto& buffer : bufferPools)
    {
        totalReservedSize += buffer->totalReservedSize();
    }
    return totalReservedSize;
}

ref_ptr<BufferInfo> MemoryBufferPools::reserveBuffer(VkDeviceSize totalSize, VkDeviceSize alignment, VkBufferUsageFlags bufferUsageFlags, VkSharingMode sharingMode, VkMemoryPropertyFlags memoryPropertiesFlags)
{
    ref_ptr<BufferInfo> bufferInfo = BufferInfo::create();

    {
        std::scoped_lock<std::mutex> lock(_mutex);
        for (auto& bufferFromPool : bufferPools)
        {
            if (bufferFromPool->usage == bufferUsageFlags && bufferFromPool->size >= totalSize)
            {
                MemorySlots::OptionalOffset reservedBufferSlot = bufferFromPool->reserve(totalSize, alignment);
                if (reservedBufferSlot.first)
                {
                    bufferInfo->buffer = bufferFromPool;
                    bufferInfo->offset = reservedBufferSlot.second;
                    bufferInfo->range = totalSize;
                    return bufferInfo;
                }
            }
        }

        VkDeviceSize deviceSize = std::max(totalSize, minimumBufferSize);

        bufferInfo->buffer = Buffer::create(deviceSize, bufferUsageFlags, sharingMode);
        bufferInfo->buffer->compile(device);

        MemorySlots::OptionalOffset reservedBufferSlot = bufferInfo->buffer->reserve(totalSize, alignment);
        bufferInfo->offset = reservedBufferSlot.second;
        bufferInfo->range = totalSize;

        //debug(name, " : Created new Buffer ", bufferInfo->buffer.get(), " totalSize ", totalSize, " deviceSize = ", deviceSize);
    }

    //debug(name, " : bufferInfo->offset = ", bufferInfo->offset);

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(*device, bufferInfo->buffer->vk(device->deviceID), &memRequirements);

    auto reservedMemorySlot = reserveMemory(memRequirements, memoryPropertiesFlags);

    if (!reservedMemorySlot.first)
    {
        debug(name, " : Failed to space for MemoryBufferPools::reserveBuffer(", totalSize, ", ", alignment, ", ", bufferUsageFlags, ")");
        //throw Exception{"Error: Failed to allocate Buffer from MemoryBufferPool.", VK_ERROR_OUT_OF_DEVICE_MEMORY};
        return {};
    }

    //debug(name, " : Allocated new buffer, MemoryBufferPools::reserveBuffer(", totalSize, ", ", alignment, ", ", bufferUsageFlags, ") ");
    bufferInfo->buffer->bind(reservedMemorySlot.first, reservedMemorySlot.second);

    //if (!bufferInfo->buffer->full())
    {
        std::scoped_lock<std::mutex> lock(_mutex);
        //debug(name, "  inserting new Buffer into Context.bufferPools");
        bufferPools.push_back(bufferInfo->buffer);
    }

    return bufferInfo;
}

MemoryBufferPools::DeviceMemoryOffset MemoryBufferPools::reserveMemory(VkMemoryRequirements memRequirements, VkMemoryPropertyFlags memoryPropertiesFlags, void* pNextAllocInfo)
{
    VkDeviceSize totalSize = memRequirements.size;
    // vsg::info("MemoryBufferPools::reserveMemory() ", totalSize, ", device->availableMemory() = ", device->availableMemory());

    std::scoped_lock<std::mutex> lock(_mutex);

    ref_ptr<DeviceMemory> deviceMemory;
    MemorySlots::OptionalOffset reservedSlot(false, 0);

    for (auto& memoryPool : memoryPools)
    {
        if (memoryPool->getMemoryRequirements().memoryTypeBits == memRequirements.memoryTypeBits &&
            memoryPool->getMemoryRequirements().alignment == memRequirements.alignment &&
            memoryPool->maximumAvailableSpace() >= totalSize)
        {
            reservedSlot = memoryPool->reserve(totalSize);
            if (reservedSlot.first)
            {
                deviceMemory = memoryPool;
                break;
            }
        }
    }

    if (!deviceMemory)
    {
        VkDeviceSize availableSpace = std::max(minimumDeviceMemorySize, totalSize);

        if (allocatedMemoryLimit < 1.0)
        {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT memoryBudget;
            memoryBudget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
            memoryBudget.pNext = nullptr;

            VkPhysicalDeviceMemoryProperties2 dmp;
            dmp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
            dmp.pNext = &memoryBudget;

            vkGetPhysicalDeviceMemoryProperties2(*(device->getPhysicalDevice()), &dmp);

            auto& memoryProperties = dmp.memoryProperties;

            for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
            {
                if ((memoryProperties.memoryTypes[i].propertyFlags & memoryPropertiesFlags) == memoryPropertiesFlags) // supported
                {
                    uint32_t heapIndex = memoryProperties.memoryTypes[i].heapIndex;

                    VkDeviceSize heapBudget = static_cast<VkDeviceSize>(static_cast<double>(memoryBudget.heapBudget[heapIndex]) * allocatedMemoryLimit);
                    VkDeviceSize heapUsage = memoryBudget.heapUsage[heapIndex];
                    VkDeviceSize heapAvailable = (heapUsage < heapBudget) ? heapBudget - heapUsage : 0;
                    availableSpace = heapAvailable;

                    break;
                }
            }

            VkDeviceSize minimumSpare = 0; //16*1024*1024;
            if (availableSpace < minimumSpare)
                availableSpace = 0;
            else
                availableSpace -= minimumSpare;
        }

        if (totalSize <= availableSpace)
        {
            if (availableSpace < minimumDeviceMemorySize)
            {
                debug("MemoryBufferPools::reserveMemory(", totalSize, ") reducing minimumDeviceMemorySize = ", minimumDeviceMemorySize, " to ", availableSpace);
                minimumDeviceMemorySize = availableSpace;
            }

            VkDeviceSize deviceMemorySize = std::max(totalSize, minimumDeviceMemorySize);

            // clamp to an aligned size
            deviceMemorySize = ((deviceMemorySize + memRequirements.alignment - 1) / memRequirements.alignment) * memRequirements.alignment;

            //debug("Creating new local DeviceMemory");
            if (memRequirements.size < deviceMemorySize) memRequirements.size = deviceMemorySize;

            deviceMemory = vsg::DeviceMemory::create(device, memRequirements, memoryPropertiesFlags, pNextAllocInfo);
            if (deviceMemory)
            {
                reservedSlot = deviceMemory->reserve(totalSize);
                // if (!deviceMemory->full())
                {
                    //debug("  inserting DeviceMemory into memoryPool ", deviceMemory.get());
                    memoryPools.push_back(deviceMemory);
                }
            }
        }
        else if (throwOutOfDeviceMemoryException)
        {
            throw vsg::Exception{"MemoryBufferPools::reserve() out of memory", VK_ERROR_OUT_OF_DEVICE_MEMORY};
        }
    }

    if (!reservedSlot.first)
    {
        debug("MemoryBufferPools::reserveMemory(", totalSize, ") failed, insufficient memory available.");
        return {};
    }

    //debug("MemoryBufferPools::reserveMemory() allocated DeviceMemoryOffset(", deviceMemory, ", ", reservedSlot.second, ")");
    return MemoryBufferPools::DeviceMemoryOffset(deviceMemory, reservedSlot.second);
}

VkResult MemoryBufferPools::reserve(ResourceRequirements& requirements)
{
    //vsg::info("MemoryBufferPools::reserve(ResourceRequirements& requirements) { ");

    auto deviceID = device->deviceID;
    const auto& limits = device->getPhysicalDevice()->getProperties().limits;

    VkMemoryPropertyFlags memoryPropertiesFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT; // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    // allocate bufferInfos
    VkDeviceSize failedBufferMemory = 0;
    for (auto& [properties, bufferInfos] : requirements.bufferInfos)
    {
        for (auto& bufferInfo : bufferInfos)
        {
            if (!bufferInfo->buffer)
            {
                VkDeviceSize alignment = 4;
                if ((properties.usageFlags & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0)
                    alignment = limits.minUniformBufferOffsetAlignment;
                else if ((properties.usageFlags & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0)
                    alignment = limits.minStorageBufferOffsetAlignment;

                debug("MemoryBufferPools::reserve(ResourceRequirements& requirements) properties.usageFlags = ", properties.usageFlags, ", alignment = ", alignment);

                auto newBufferInfo = reserveBuffer(bufferInfo->data->dataSize(), alignment, properties.usageFlags, properties.sharingMode, memoryPropertiesFlags);
                if (newBufferInfo)
                {
                    bufferInfo->take(*newBufferInfo);

                    //info("    ALLOCATED usage = ", properties.usageFlags, ", alignment = ", alignment, ", bufferInfo->data = ", bufferInfo->data, ", offset = ", bufferInfo->offset, ", range = ", bufferInfo->range, ", buffer = ", bufferInfo->buffer, " ----- size = ", computeSize(*bufferInfo));
                }
                else
                {
                    failedBufferMemory += bufferInfo->computeDataSize();
                }
            }
        }
    }

    // allocate images
    VkDeviceSize failedImageMemory = 0;
    for (auto& imageInfo : requirements.imageInfos)
    {
        if (imageInfo->imageView && imageInfo->imageView->image && imageInfo->imageView->image->getDeviceMemory(deviceID) == 0)
        {
            if (imageInfo->imageView->image->compile(*this) == VK_SUCCESS && imageInfo->imageView->image->getDeviceMemory(deviceID) != 0)
            {
                //info("    ALLOCATED imageInfo = ", imageInfo, ", imageView = ", imageInfo->imageView, " ----- size = ", computeSize(*imageInfo), " device memory = ", imageInfo->imageView->image->getDeviceMemory(deviceID), " offset = ", imageInfo->imageView->image->getMemoryOffset(deviceID));
            }
            else
            {
                failedImageMemory += imageInfo->computeDataSize();
            }
        }
    }

    VkDeviceSize memoryRequired = failedBufferMemory + failedImageMemory;

    // all required resources allocated
    if (memoryRequired == 0)
    {
        //info("MemoryBufferPools::reserve() memoryRequired = ", memoryRequired);
        return VK_SUCCESS;
    }
    else
    {
        //VkDeviceSize memoryAvailable = device->availableMemory(false);
        //info("MemoryBufferPools::reserve() out of memory: memoryAvailable = ", memoryAvailable, " memoryRequired = ", memoryRequired);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
}

void MemoryBufferPools::report(LogOutput& out) const
{
    out.enter("MemoryBufferPools::report(..)");
    for(const auto& memoryPool : memoryPools)
    {
        memoryPool->report(out);
    }
    out.leave();
}
