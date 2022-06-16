#pragma once

#include <optional>
#include <unordered_map>
#include <vector>

#include "dxvk_hash.h"
#include "dxvk_include.h"

namespace dxvk {

  class DxvkDevice;

  /**
   * \brief Descriptor set indices
   */
  struct DxvkDescriptorSets {
    static constexpr uint32_t CsAll       = 0;
    static constexpr uint32_t FsViews     = 0;
    static constexpr uint32_t FsBuffers   = 1;
    static constexpr uint32_t VsAll       = 2;
    static constexpr uint32_t SetCount    = 3;
  };

  /**
   * \brief Binding info
   *
   * Stores metadata for a single binding in
   * a given shader, or for the whole pipeline.
   */
  struct DxvkBindingInfo {
    VkDescriptorType    descriptorType;   ///< Vulkan descriptor type
    uint32_t            resourceBinding;  ///< API binding slot for the resource
    VkImageViewType     viewType;         ///< Image view type
    VkShaderStageFlags  stages;           ///< Shader stage mask
    VkAccessFlags       access;           ///< Access mask for the resource

    /**
     * \brief Computes descriptor set index for the given binding
     *
     * This is determines based on the shader stages that use the binding.
     * \returns Descriptor set index
     */
    uint32_t computeSetIndex() const;

    /**
     * \brief Checks whether bindings can be merged
     *
     * Bindings can be merged if they access the same resource with
     * the same view and descriptor type and are part of the same
     * descriptor set.
     * \param [in] binding The binding to probe
     * \returns \c true if the bindings can be merged
     */
    bool canMerge(const DxvkBindingInfo& binding) const;

    /**
     * \brief Merges bindings
     *
     * Merges the stage and access flags of two
     * otherwise identical binding declarations.
     * \param [in] binding The binding to merge
     */
    void merge(const DxvkBindingInfo& binding);

    /**
     * \brief Checks for equality
     *
     * \param [in] other Binding to compare to
     * \returns \c true if both bindings are equal
     */
    bool eq(const DxvkBindingInfo& other) const;

    /**
     * \brief Hashes binding info
     * \returns Binding hash
     */
    size_t hash() const;

  };

  /**
   * \brief Binding layout
   *
   * Convenience class to map out shader bindings for use in
   * descriptor set layouts and pipeline layouts. If possible,
   * bindings that only differ in stage will be merged.
   */
  class DxvkBindingLayout {

  public:

    DxvkBindingLayout();
    ~DxvkBindingLayout();

    /**
     * \brief Number of Vulkan bindings per set
     *
     * \param [in] set Descriptor set index
     * \returns Binding count for the given set
     */
    uint32_t getBindingCount(uint32_t set) const {
      return uint32_t(m_bindings[set].size());
    }

    /**
     * \brief Retrieves mapped binding info
     *
     * \param [in] set Descriptor set index
     * \param [in] idx Binding index
     * \returns Binding info
     */
    const DxvkBindingInfo& getBinding(uint32_t set, uint32_t idx) const {
      return m_bindings[set][idx];
    }

    /**
     * \brief Retrieves push constant range
     * \returns Push constant range
     */
    VkPushConstantRange getPushConstantRange() const {
      return m_pushConst;
    }

    /**
     * \brief Adds a binding to the layout
     *
     * \param [in] binding Binding info
     * \brief Set and binding index
     */
    void addBinding(const DxvkBindingInfo& binding);

    /**
     * \brief Adds push constant range
     * \param [in] range Push constant range
     */
    void addPushConstantRange(VkPushConstantRange range);

    /**
     * \brief Merges binding layouts
     *
     * Adds bindings and push constant range from another layout to
     * the current layout. Useful when creating pipeline layouts and
     * descriptor set layouts for pipelines consisting of multiple
     * shader stages.
     * Note that merging layouts can change Vulkan binding numbers.
     * \param [in] layout Binding layout to merge
     */
    void merge(const DxvkBindingLayout& layout);

    /**
     * \brief Checks for equality
     *
     * \param [in] other Binding layout to compare to
     * \returns \c true if both binding layouts are equal
     */
    bool eq(const DxvkBindingLayout& other) const;

    /**
     * \brief Hashes binding layout
     * \returns Binding layout hash
     */
    size_t hash() const;

  private:

    std::array<std::vector<DxvkBindingInfo>, DxvkDescriptorSets::SetCount> m_bindings;
    VkPushConstantRange                                                    m_pushConst;

  };

  /**
   * \brief Descriptor set and binding number
   */
  struct DxvkBindingMapping {
    uint32_t set;
    uint32_t binding;
    uint32_t constId;
  };

  /**
   * \brief Pipeline and descriptor set layouts for a given binding layout
   *
   * Creates the following Vulkan objects for a given binding layout:
   * - A descriptor set layout for each required descriptor set
   * - A descriptor update template for each set with non-zero binding count
   * - A pipeline layout referencing all descriptor sets and the push constant ranges
   */
  class DxvkBindingLayoutObjects {

  public:

    DxvkBindingLayoutObjects(
            DxvkDevice*             device,
      const DxvkBindingLayout&      layout);

    ~DxvkBindingLayoutObjects();

    /**
     * \brief Binding layout
     * \returns Binding layout
     */
    const DxvkBindingLayout& layout() const {
      return m_layout;
    }

    /**
     * \brief Queries active descriptor set mask
     * \returns Bit mask of non-empty descriptor sets
     */
    uint32_t getSetMask() const {
      return m_setMask;
    }

    /**
     * \brief Queries first binding number for a given set
     *
     * This is relevant for generating binding masks.
     * \param [in] set Descriptor set index
     * \returns First binding in the given set
     */
    uint32_t getFirstBinding(uint32_t set) const {
      return m_bindingOffsets[set];
    }

    /**
     * \brief Retrieves descriptor set layout for a given set
     *
     * \param [in] set Descriptor set index
     * \returns Vulkan descriptor set layout
     */
    VkDescriptorSetLayout getSetLayout(uint32_t set) const {
      return m_setLayouts[set];
    }

    /**
     * \brief Retrieves descriptor update template for a given set
     *
     * \param [in] set Descriptor set index
     * \returns Vulkan descriptor update template
     */
    VkDescriptorUpdateTemplate getSetUpdateTemplate(uint32_t set) const {
      return m_setTemplates[set];
    }

    /**
     * \brief Retrieves pipeline layout
     * \returns Pipeline layout
     */
    VkPipelineLayout getPipelineLayout() const {
      return m_pipelineLayout;
    }

    /**
     * \brief Looks up set and binding number by resource binding
     *
     * \param [in] index Resource binding index
     * \returns Descriptor set and binding number
     */
    std::optional<DxvkBindingMapping> lookupBinding(uint32_t index) const {
      auto entry = m_mapping.find(index);

      if (entry != m_mapping.end())
        return entry->second;

      return std::nullopt;
    }

    /**
     * \brief Queries accumulated resource access flags
     *
     * Can be used to determine whether the pipeline
     * reads or writes any resources.
     */
    VkAccessFlags getAccessFlags() const;

  private:

    DxvkDevice*         m_device;
    DxvkBindingLayout   m_layout;
    VkPipelineLayout    m_pipelineLayout  = VK_NULL_HANDLE;

    uint32_t            m_setMask         = 0;

    std::array<VkDescriptorSetLayout,       DxvkDescriptorSets::SetCount> m_setLayouts   = { };
    std::array<VkDescriptorUpdateTemplate,  DxvkDescriptorSets::SetCount> m_setTemplates = { };
    std::array<uint32_t,                    DxvkDescriptorSets::SetCount> m_bindingOffsets = { };

    std::unordered_map<uint32_t, DxvkBindingMapping> m_mapping;

  };


  /**
   * \brief Dirty descriptor set state
   */
  class DxvkDescriptorState {

  public:

    void dirtyBuffers(VkShaderStageFlags stages) {
      m_dirtyBuffers  |= stages;
    }

    void dirtyViews(VkShaderStageFlags stages) {
      m_dirtyViews    |= stages;
    }

    void dirtyStages(VkShaderStageFlags stages) {
      m_dirtyBuffers  |= stages;
      m_dirtyViews    |= stages;
    }

    void clearStages(VkShaderStageFlags stages) {
      m_dirtyBuffers  &= ~stages;
      m_dirtyViews    &= ~stages;
    }

    bool hasDirtyGraphicsSets() const {
      return (m_dirtyBuffers | m_dirtyViews) & (VK_SHADER_STAGE_ALL_GRAPHICS);
    }

    bool hasDirtyComputeSets() const {
      return (m_dirtyBuffers | m_dirtyViews) & (VK_SHADER_STAGE_COMPUTE_BIT);
    }

    uint32_t getDirtyGraphicsSets() const {
      uint32_t result = 0;
      if (m_dirtyBuffers & VK_SHADER_STAGE_FRAGMENT_BIT)
        result |= (1u << DxvkDescriptorSets::FsBuffers);
      if (m_dirtyViews & VK_SHADER_STAGE_FRAGMENT_BIT)
        result |= (1u << DxvkDescriptorSets::FsViews) | (1u << DxvkDescriptorSets::FsBuffers);
      if ((m_dirtyBuffers | m_dirtyViews) & (VK_SHADER_STAGE_ALL_GRAPHICS & ~VK_SHADER_STAGE_FRAGMENT_BIT))
        result |= (1u << DxvkDescriptorSets::VsAll);
      return result;
    }

    uint32_t getDirtyComputeSets() const {
      uint32_t result = 0;
      if ((m_dirtyBuffers | m_dirtyViews) & (VK_SHADER_STAGE_COMPUTE_BIT))
        result |= (1u << DxvkDescriptorSets::CsAll);
      return result;
    }

    void clearSets() {
      for (size_t i = 0; i < m_sets.size(); i++)
        m_sets[i] = VK_NULL_HANDLE;
    }

    template<VkPipelineBindPoint BindPoint>
    VkDescriptorSet& getSet(uint32_t index) {
      return m_sets[BindPoint * DxvkDescriptorSets::SetCount + index];
    }

    template<VkPipelineBindPoint BindPoint>
    const VkDescriptorSet& getSet(uint32_t index) const {
      return m_sets[BindPoint * DxvkDescriptorSets::SetCount + index];
    }

  private:

    VkShaderStageFlags m_dirtyBuffers   = 0;
    VkShaderStageFlags m_dirtyViews     = 0;

    std::array<VkDescriptorSet, 2 * DxvkDescriptorSets::SetCount> m_sets;

  };


  /**
   * \brief Resource slot
   * 
   * Describes the type of a single resource
   * binding that a shader can access.
   */
  struct DxvkResourceSlot {
    uint32_t           slot;
    VkDescriptorType   type;
    VkImageViewType    view;
    VkAccessFlags      access;
  };
  
  /**
   * \brief Shader interface binding
   * 
   * Corresponds to a single descriptor binding in
   * Vulkan. DXVK does not use descriptor arrays.
   * Instead, each binding stores one descriptor.
   */
  struct DxvkDescriptorSlot {
    uint32_t           slot;    ///< Resource slot index for the context
    VkDescriptorType   type;    ///< Descriptor type (aka resource type)
    VkImageViewType    view;    ///< Compatible image view type
    VkShaderStageFlags stages;  ///< Stages that can use the resource
    VkAccessFlags      access;  ///< Access flags
  };
  
  
  /**
   * \brief Descriptor slot mapping
   * 
   * Convenience class that generates descriptor slot
   * index to binding index mappings. This is required
   * when generating Vulkan pipeline and descriptor set
   * layouts.
   */
  class DxvkDescriptorSlotMapping {
    constexpr static uint32_t InvalidBinding = 0xFFFFFFFFu;
  public:
    
    DxvkDescriptorSlotMapping();
    ~DxvkDescriptorSlotMapping();
    
    /**
     * \brief Number of descriptor bindings
     * \returns Descriptor binding count
     */
    uint32_t bindingCount() const {
      return m_descriptorSlots.size();
    }
    
    /**
     * \brief Descriptor binding infos
     * \returns Descriptor binding infos
     */
    const DxvkDescriptorSlot* bindingInfos() const {
      return m_descriptorSlots.data();
    }

    /**
     * \brief Push constant range
     * \returns Push constant range
     */
    VkPushConstantRange pushConstRange() const {
      return m_pushConstRange;
    }
    
    /**
     * \brief Defines a new slot
     * 
     * Adds a slot to the mapping. If the slot is already
     * defined by another shader stage, this will extend
     * the stage mask by the given stage. Otherwise, an
     * entirely new binding is added.
     * \param [in] stage Shader stage
     * \param [in] desc Slot description
     */
    void defineSlot(
            VkShaderStageFlagBits stage,
      const DxvkResourceSlot&     desc);

    /**
     * \brief Defines new push constant range
     *
     * \param [in] stage Shader stage
     * \param [in] offset Range offset
     * \param [in] size Range size
     */
    void definePushConstRange(
            VkShaderStageFlagBits stage,
            uint32_t              offset,
            uint32_t              size);
    
    /**
     * \brief Gets binding ID for a slot
     * 
     * \param [in] slot Resource slot
     * \returns Binding index, or \c InvalidBinding
     */
    uint32_t getBindingId(
            uint32_t              slot) const;
    
    /**
     * \brief Makes static descriptors dynamic
     * 
     * Replaces static uniform and storage buffer bindings by
     * their dynamic equivalent if the number of bindings of
     * the respective type lies within supported device limits.
     * Using dynamic descriptor types may improve performance.
     * \param [in] uniformBuffers Max number of uniform buffers
     * \param [in] storageBuffers Max number of storage buffers
     */
    void makeDescriptorsDynamic(
            uint32_t              uniformBuffers,
            uint32_t              storageBuffers);
    
  private:
    
    std::vector<DxvkDescriptorSlot> m_descriptorSlots;
    VkPushConstantRange             m_pushConstRange = { };

    uint32_t countDescriptors(
            VkDescriptorType      type) const;

    void replaceDescriptors(
            VkDescriptorType      oldType,
            VkDescriptorType      newType);
    
  };
  
  
  /**
   * \brief Shader interface
   * 
   * Describes shader resource bindings
   * for a graphics or compute pipeline.
   */
  class DxvkPipelineLayout : public RcObject {
    
  public:
    
    DxvkPipelineLayout(
      const Rc<vk::DeviceFn>&   vkd,
      const DxvkDescriptorSlotMapping& slotMapping,
            VkPipelineBindPoint pipelineBindPoint);
    
    ~DxvkPipelineLayout();
    
    /**
     * \brief Number of resource bindings
     * \returns Resource binding count
     */
    uint32_t bindingCount() const {
      return m_bindingSlots.size();
    }
    
    /**
     * \brief Resource binding info
     * 
     * \param [in] id Binding index
     * \returns Resource binding info
     */
    const DxvkDescriptorSlot& binding(uint32_t id) const {
      return m_bindingSlots[id];
    }
    
    /**
     * \brief Resource binding info
     * \returns Resource binding info
     */
    const DxvkDescriptorSlot* bindings() const {
      return m_bindingSlots.data();
    }
    
    /**
     * \brief Push constant range
     * \returns Push constant range
     */
    const VkPushConstantRange& pushConstRange() const {
      return m_pushConstRange;
    }
    
    /**
     * \brief Descriptor set layout handle
     * \returns Descriptor set layout handle
     */
    VkDescriptorSetLayout descriptorSetLayout() const {
      return m_descriptorSetLayout;
    }
    
    /**
     * \brief Pipeline layout handle
     * \returns Pipeline layout handle
     */
    VkPipelineLayout pipelineLayout() const {
      return m_pipelineLayout;
    }
    
    /**
     * \brief Descriptor update template
     * \returns Descriptor update template
     */
    VkDescriptorUpdateTemplateKHR descriptorTemplate() const {
      return m_descriptorTemplate;
    }

    /**
     * \brief Number of dynamic bindings
     * \returns Dynamic binding count
     */
    uint32_t dynamicBindingCount() const {
      return m_dynamicSlots.size();
    }

    /**
     * \brief Returns a dynamic binding
     * 
     * \param [in] id Dynamic binding ID
     * \returns Reference to that binding
     */
    const DxvkDescriptorSlot& dynamicBinding(uint32_t id) const {
      return this->binding(m_dynamicSlots[id]);
    }
    
    /**
     * \brief Checks for static buffer bindings
     * 
     * Returns \c true if there is at least one
     * descriptor of the static uniform buffer
     * type.
     */
    bool hasStaticBufferBindings() const {
      return m_descriptorTypes.test(
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    }
    
    /**
     * \brief Checks whether buffers or images are written to
     * 
     * It is assumed that storage images and buffers
     * will be written to if they are present. Used
     * for synchronization purposes.
     * \param [in] stages Shader stages to check
     */
    VkShaderStageFlags getStorageDescriptorStages() const {
      VkShaderStageFlags stages = 0;

      for (const auto& slot : m_bindingSlots) {
        if (slot.access & VK_ACCESS_SHADER_WRITE_BIT)
          stages |= slot.stages;
      }
      
      return stages;
    }

  private:
    
    Rc<vk::DeviceFn> m_vkd;
    
    VkPushConstantRange             m_pushConstRange      = { };
    VkDescriptorSetLayout           m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout                m_pipelineLayout      = VK_NULL_HANDLE;
    VkDescriptorUpdateTemplateKHR   m_descriptorTemplate  = VK_NULL_HANDLE;
    
    std::vector<DxvkDescriptorSlot> m_bindingSlots;
    std::vector<uint32_t>           m_dynamicSlots;

    Flags<VkDescriptorType>         m_descriptorTypes;
    
  };
  
}