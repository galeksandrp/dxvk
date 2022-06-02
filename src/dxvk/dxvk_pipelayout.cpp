#include <cstring>

#include "dxvk_device.h"
#include "dxvk_descriptor.h"
#include "dxvk_limits.h"
#include "dxvk_pipelayout.h"

namespace dxvk {
  
  uint32_t DxvkBindingInfo::computeSetIndex() const {
    if (stages & VK_SHADER_STAGE_COMPUTE_BIT) {
      // Put all compute shader resources into a single set
      return DxvkDescriptorSets::CsAll;
    } else if (stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
      // For fragment shaders, create a separate set for UBOs
      if (descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
       || descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        return DxvkDescriptorSets::FsBuffers;

      return DxvkDescriptorSets::FsViews;
    } else {
      // Put all vertex shader resources into the last set.
      // Vertex shader UBOs are usually updated every draw,
      // and other resource types are rarely used.
      return DxvkDescriptorSets::VsAll;
    }
  }


  bool DxvkBindingInfo::canMerge(const DxvkBindingInfo& binding) const {
    if ((stages & VK_SHADER_STAGE_FRAGMENT_BIT) != (binding.stages & VK_SHADER_STAGE_FRAGMENT_BIT))
      return false;

    return descriptorType    == binding.descriptorType
        && resourceBinding   == binding.resourceBinding
        && viewType          == binding.viewType;
  }


  void DxvkBindingInfo::merge(const DxvkBindingInfo& binding) {
    stages |= binding.stages;
    access |= binding.access;
  }


  bool DxvkBindingInfo::eq(const DxvkBindingInfo& other) const {
    return descriptorType    == other.descriptorType
        && resourceBinding   == other.resourceBinding
        && viewType          == other.viewType
        && stages            == other.stages
        && access            == other.access;
  }


  size_t DxvkBindingInfo::hash() const {
    DxvkHashState hash;
    hash.add(descriptorType);
    hash.add(resourceBinding);
    hash.add(viewType);
    hash.add(stages);
    hash.add(access);
    return hash;
  }


  DxvkBindingLayout::DxvkBindingLayout()
  : m_pushConst { 0, 0, 0 } {

  }


  DxvkBindingLayout::~DxvkBindingLayout() {

  }


  void DxvkBindingLayout::addBinding(const DxvkBindingInfo& binding) {
    uint32_t set = binding.computeSetIndex();

    for (auto& b : m_bindings[set]) {
      if (b.canMerge(binding)) {
        b.merge(binding);
        return;
      }
    }

    m_bindings[set].push_back(binding);
  }


  void DxvkBindingLayout::addPushConstantRange(VkPushConstantRange range) {
    uint32_t oldEnd = m_pushConst.offset + m_pushConst.size;
    uint32_t newEnd = range.offset + range.size;

    m_pushConst.stageFlags |= range.stageFlags;
    m_pushConst.offset = std::min(m_pushConst.offset, range.offset);
    m_pushConst.size = std::max(oldEnd, newEnd) - m_pushConst.offset;
  }


  void DxvkBindingLayout::merge(const DxvkBindingLayout& layout) {
    for (uint32_t i = 0; i < layout.m_bindings.size(); i++) {
      for (const auto& binding : layout.m_bindings[i])
        addBinding(binding);
    }

    addPushConstantRange(layout.m_pushConst);
  }


  bool DxvkBindingLayout::eq(const DxvkBindingLayout& other) const {
    for (uint32_t i = 0; i < m_bindings.size(); i++) {
      if (m_bindings[i].size() != other.m_bindings[i].size())
        return false;
    }

    for (uint32_t i = 0; i < m_bindings.size(); i++) {
      for (uint32_t j = 0; j < m_bindings[i].size(); j++) {
        if (!m_bindings[i][j].eq(other.m_bindings[i][j]))
          return false;
      }
    }

    if (m_pushConst.stageFlags != other.m_pushConst.stageFlags
     || m_pushConst.offset     != other.m_pushConst.offset
     || m_pushConst.size       != other.m_pushConst.size)
      return false;

    return true;
  }


  size_t DxvkBindingLayout::hash() const {
    DxvkHashState hash;

    for (uint32_t i = 0; i < m_bindings.size(); i++) {
      for (uint32_t j = 0; j < m_bindings[i].size(); j++)
        hash.add(m_bindings[i][j].hash());
    }

    hash.add(m_pushConst.stageFlags);
    hash.add(m_pushConst.offset);
    hash.add(m_pushConst.size);
    return hash;
  }


  DxvkBindingLayoutObjects::DxvkBindingLayoutObjects(
          DxvkDevice*             device,
    const DxvkBindingLayout&      layout)
  : m_device(device), m_layout(layout) {
    auto vk = m_device->vkd();

    std::array<VkDescriptorSetLayoutBinding, MaxNumActiveBindings> bindingInfos;
    std::array<VkDescriptorUpdateTemplateEntry, MaxNumActiveBindings> templateInfos;

    uint32_t constId = 0;

    for (uint32_t i = 0; i < DxvkDescriptorSets::SetCount; i++) {
      m_bindingOffsets[i] = constId;

      VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
      layoutInfo.bindingCount = m_layout.getBindingCount(i);
      layoutInfo.pBindings = bindingInfos.data();

      for (uint32_t j = 0; j < layoutInfo.bindingCount; j++) {
        const DxvkBindingInfo& binding = m_layout.getBinding(i, j);

        VkDescriptorSetLayoutBinding& bindingInfo = bindingInfos[j];
        bindingInfo.binding = j;
        bindingInfo.descriptorType = binding.descriptorType;
        bindingInfo.descriptorCount = 1;
        bindingInfo.stageFlags = binding.stages;
        bindingInfo.pImmutableSamplers = nullptr;

        VkDescriptorUpdateTemplateEntry& templateInfo = templateInfos[j];
        templateInfo.dstBinding = j;
        templateInfo.dstArrayElement = 0;
        templateInfo.descriptorCount = 1;
        templateInfo.descriptorType = binding.descriptorType;
        templateInfo.offset = sizeof(DxvkDescriptorInfo) * j;
        templateInfo.stride = sizeof(DxvkDescriptorInfo);

        DxvkBindingMapping mapping;
        mapping.set = i;
        mapping.binding = j;
        mapping.constId = constId++;

        m_mapping.insert({ binding.resourceBinding, mapping });
      }

      if (vk->vkCreateDescriptorSetLayout(vk->device(), &layoutInfo, nullptr, &m_setLayouts[i]) != VK_SUCCESS)
        throw DxvkError("DxvkBindingLayoutObjects: Failed to create descriptor set layout");

      if (layoutInfo.bindingCount) {
        VkDescriptorUpdateTemplateCreateInfo templateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO };
        templateInfo.descriptorUpdateEntryCount = layoutInfo.bindingCount;
        templateInfo.pDescriptorUpdateEntries = templateInfos.data();
        templateInfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
        templateInfo.descriptorSetLayout = m_setLayouts[i];

        if (vk->vkCreateDescriptorUpdateTemplate(vk->device(), &templateInfo, nullptr, &m_setTemplates[i]) != VK_SUCCESS)
          throw DxvkError("DxvkBindingLayoutObjects: Failed to create descriptor update template");

        m_setMask |= 1u << i;
      }
    }

    VkPushConstantRange pushConst = m_layout.getPushConstantRange();

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutInfo.setLayoutCount = m_setLayouts.size();
    pipelineLayoutInfo.pSetLayouts = m_setLayouts.data();

    if (pushConst.stageFlags && pushConst.size) {
      pipelineLayoutInfo.pushConstantRangeCount = 1;
      pipelineLayoutInfo.pPushConstantRanges = &pushConst;
    }

    if (vk->vkCreatePipelineLayout(vk->device(), &pipelineLayoutInfo, nullptr, &m_pipelineLayout))
      throw DxvkError("DxvkBindingLayoutObjects: Failed to create pipeline layout");
  }


  DxvkBindingLayoutObjects::~DxvkBindingLayoutObjects() {
    auto vk = m_device->vkd();

    vk->vkDestroyPipelineLayout(vk->device(), m_pipelineLayout, nullptr);

    for (uint32_t i = 0; i < DxvkDescriptorSets::SetCount; i++) {
      vk->vkDestroyDescriptorSetLayout(vk->device(), m_setLayouts[i], nullptr);
      vk->vkDestroyDescriptorUpdateTemplate(vk->device(), m_setTemplates[i], nullptr);
    }
  }


  VkAccessFlags DxvkBindingLayoutObjects::getAccessFlags() const {
    VkAccessFlags flags = 0;

    for (uint32_t i = 0; i < DxvkDescriptorSets::SetCount; i++) {
      for (uint32_t j = 0; j < m_layout.getBindingCount(i); j++)
        flags |= m_layout.getBinding(i, j).access;
    }

    return flags;
  }


  DxvkDescriptorSlotMapping:: DxvkDescriptorSlotMapping() { }
  DxvkDescriptorSlotMapping::~DxvkDescriptorSlotMapping() { }
  
  
  void DxvkDescriptorSlotMapping::defineSlot(
          VkShaderStageFlagBits stage,
    const DxvkResourceSlot&     desc) {
    uint32_t bindingId = this->getBindingId(desc.slot);
    
    if (bindingId != InvalidBinding) {
      m_descriptorSlots[bindingId].stages |= stage;
      m_descriptorSlots[bindingId].access |= desc.access;
    } else {
      DxvkDescriptorSlot slotInfo;
      slotInfo.slot   = desc.slot;
      slotInfo.type   = desc.type;
      slotInfo.view   = desc.view;
      slotInfo.stages = stage;
      slotInfo.access = desc.access;
      m_descriptorSlots.push_back(slotInfo);
    }
  }


  void DxvkDescriptorSlotMapping::definePushConstRange(
          VkShaderStageFlagBits stage,
          uint32_t              offset,
          uint32_t              size) {
    m_pushConstRange.stageFlags |= stage;
    m_pushConstRange.size = std::max(
      m_pushConstRange.size, offset + size);
  }
  
  
  uint32_t DxvkDescriptorSlotMapping::getBindingId(uint32_t slot) const {
    // This won't win a performance competition, but the number
    // of bindings used by a shader is usually much smaller than
    // the number of resource slots available to the system.
    for (uint32_t i = 0; i < m_descriptorSlots.size(); i++) {
      if (m_descriptorSlots[i].slot == slot)
        return i;
    }
    
    return InvalidBinding;
  }
  
  
  void DxvkDescriptorSlotMapping::makeDescriptorsDynamic(
          uint32_t              uniformBuffers,
          uint32_t              storageBuffers) {
    if (this->countDescriptors(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) <= uniformBuffers)
      this->replaceDescriptors(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
  }


  uint32_t DxvkDescriptorSlotMapping::countDescriptors(
          VkDescriptorType      type) const {
    uint32_t count = 0;

    for (const auto& slot : m_descriptorSlots)
      count += slot.type == type ? 1 : 0;
    
    return count;
  }


  void DxvkDescriptorSlotMapping::replaceDescriptors(
          VkDescriptorType      oldType,
          VkDescriptorType      newType) {
    for (auto& slot : m_descriptorSlots) {
      if (slot.type == oldType)
        slot.type = newType;
    }
  }


  DxvkPipelineLayout::DxvkPipelineLayout(
    const Rc<vk::DeviceFn>&   vkd,
    const DxvkDescriptorSlotMapping& slotMapping,
          VkPipelineBindPoint pipelineBindPoint)
  : m_vkd           (vkd),
    m_pushConstRange(slotMapping.pushConstRange()),
    m_bindingSlots  (slotMapping.bindingCount()) {

    auto bindingCount = slotMapping.bindingCount();
    auto bindingInfos = slotMapping.bindingInfos();
    
    if (bindingCount > MaxNumActiveBindings)
      throw DxvkError(str::format("Too many active bindings in pipeline layout (", bindingCount, ")"));
    
    for (uint32_t i = 0; i < bindingCount; i++)
      m_bindingSlots[i] = bindingInfos[i];
    
    std::vector<VkDescriptorSetLayoutBinding>    bindings(bindingCount);
    std::vector<VkDescriptorUpdateTemplateEntry> tEntries(bindingCount);
    
    for (uint32_t i = 0; i < bindingCount; i++) {
      bindings[i].binding            = i;
      bindings[i].descriptorType     = bindingInfos[i].type;
      bindings[i].descriptorCount    = 1;
      bindings[i].stageFlags         = bindingInfos[i].stages;
      bindings[i].pImmutableSamplers = nullptr;
      
      tEntries[i].dstBinding      = i;
      tEntries[i].dstArrayElement = 0;
      tEntries[i].descriptorCount = 1;
      tEntries[i].descriptorType  = bindingInfos[i].type;
      tEntries[i].offset          = sizeof(DxvkDescriptorInfo) * i;
      tEntries[i].stride          = 0;

      if (bindingInfos[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
        m_dynamicSlots.push_back(i);
      
      m_descriptorTypes.set(bindingInfos[i].type);
    }
    
    // Create descriptor set layout. We do not need to
    // create one if there are no active resource bindings.
    if (bindingCount > 0) {
      VkDescriptorSetLayoutCreateInfo dsetInfo;
      dsetInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      dsetInfo.pNext        = nullptr;
      dsetInfo.flags        = 0;
      dsetInfo.bindingCount = bindings.size();
      dsetInfo.pBindings    = bindings.data();
      
      if (m_vkd->vkCreateDescriptorSetLayout(m_vkd->device(),
            &dsetInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS)
        throw DxvkError("DxvkPipelineLayout: Failed to create descriptor set layout");
    }
    
    // Create pipeline layout with the given descriptor set layout
    VkPipelineLayoutCreateInfo pipeInfo;
    pipeInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeInfo.pNext                  = nullptr;
    pipeInfo.flags                  = 0;
    pipeInfo.setLayoutCount         = bindingCount > 0 ? 1 : 0;
    pipeInfo.pSetLayouts            = &m_descriptorSetLayout;
    pipeInfo.pushConstantRangeCount = 0;
    pipeInfo.pPushConstantRanges    = nullptr;

    if (m_pushConstRange.size) {
      pipeInfo.pushConstantRangeCount = 1;
      pipeInfo.pPushConstantRanges    = &m_pushConstRange;
    }
    
    if (m_vkd->vkCreatePipelineLayout(m_vkd->device(),
        &pipeInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
      m_vkd->vkDestroyDescriptorSetLayout(m_vkd->device(), m_descriptorSetLayout, nullptr);
      throw DxvkError("DxvkPipelineLayout: Failed to create pipeline layout");
    }
    
    // Create descriptor update template. If there are no active
    // resource bindings, there won't be any descriptors to update.
    if (bindingCount > 0) {
      VkDescriptorUpdateTemplateCreateInfo templateInfo;
      templateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
      templateInfo.pNext = nullptr;
      templateInfo.flags = 0;
      templateInfo.descriptorUpdateEntryCount = tEntries.size();
      templateInfo.pDescriptorUpdateEntries   = tEntries.data();
      templateInfo.templateType               = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
      templateInfo.descriptorSetLayout        = m_descriptorSetLayout;
      templateInfo.pipelineBindPoint          = pipelineBindPoint;
      templateInfo.pipelineLayout             = m_pipelineLayout;
      templateInfo.set                        = 0;
      
      if (m_vkd->vkCreateDescriptorUpdateTemplate(
          m_vkd->device(), &templateInfo, nullptr, &m_descriptorTemplate) != VK_SUCCESS) {
        m_vkd->vkDestroyDescriptorSetLayout(m_vkd->device(), m_descriptorSetLayout, nullptr);
        m_vkd->vkDestroyPipelineLayout(m_vkd->device(), m_pipelineLayout, nullptr);
        throw DxvkError("DxvkPipelineLayout: Failed to create descriptor update template");
      }
    }
  }
  
  
  DxvkPipelineLayout::~DxvkPipelineLayout() {
    m_vkd->vkDestroyDescriptorUpdateTemplate(
      m_vkd->device(), m_descriptorTemplate, nullptr);
    
    m_vkd->vkDestroyPipelineLayout(
      m_vkd->device(), m_pipelineLayout, nullptr);
    
    m_vkd->vkDestroyDescriptorSetLayout(
      m_vkd->device(), m_descriptorSetLayout, nullptr);
  }
  
}