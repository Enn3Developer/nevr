use std::sync::Arc;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::device::Device;
use vulkano::pipeline::PipelineLayout;
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::shader::ShaderStages;

pub struct VulkanDescriptorBinding {
    pub stage: ShaderStages,
    pub descriptor_type: DescriptorType,
}

pub struct VulkanDescriptorSet<'a> {
    pub bindings: &'a [VulkanDescriptorBinding],
}

pub fn new_pipeline_layout(
    device: Arc<Device>,
    descriptor_sets: &[VulkanDescriptorSet],
) -> Arc<PipelineLayout> {
    PipelineLayout::new(
        device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: descriptor_sets
                .into_iter()
                .map(|descriptor_set| {
                    DescriptorSetLayout::new(
                        device.clone(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: descriptor_set
                                .bindings
                                .into_iter()
                                .enumerate()
                                .map(|(id, binding)| {
                                    (
                                        id as u32,
                                        DescriptorSetLayoutBinding {
                                            stages: binding.stage,
                                            ..DescriptorSetLayoutBinding::descriptor_type(
                                                binding.descriptor_type,
                                            )
                                        },
                                    )
                                })
                                .collect(),
                            ..Default::default()
                        },
                    )
                    .unwrap()
                })
                .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}
